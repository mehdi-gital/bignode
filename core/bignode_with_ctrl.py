import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev
from torch_geometric.nn import aggr
from torchdiffeq import odeint


class BoundaryInjectedMessagePassingLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            control_dim,
            message_dim,
            hidden_dim,
            interior_node_index,
            boundary_node_index,
            control_node_index
    ):
        super(BoundaryInjectedMessagePassingLayer, self).__init__()
        # node indices for interior, boundary and control
        self.interior_node_index = interior_node_index
        self.boundary_node_index = boundary_node_index
        self.control_node_index = control_node_index

        # message functions for {interior, boundary, control} -> interior
        self.message_int_int = nn.Linear(in_features=2 * input_dim, out_features=message_dim)
        self.message_bound_int = nn.Linear(in_features=2 * input_dim, out_features=message_dim)
        self.message_ctrl_int = nn.Linear(in_features=input_dim + control_dim, out_features=message_dim)

        # message functions for {boundary} -> boundary
        self.message_bound_bound = nn.Linear(in_features=2 * input_dim, out_features=message_dim)

        # message functions for {control} -> control
        self.message_ctrl_ctrl = nn.Linear(in_features=2 * control_dim, out_features=message_dim)

        # message aggregator function
        self.message_aggregator = aggr.MeanAggregation()

        # SageConv for update functions:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html

        # interior update
        self.interior_msg_W = nn.Linear(in_features=message_dim, out_features=hidden_dim)
        self.interior_self_W = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        # boundary update
        self.boundary_msg_W = nn.Linear(in_features=message_dim, out_features=hidden_dim)
        self.boundary_self_W = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        # control update
        self.control_msg_W = nn.Linear(in_features=message_dim, out_features=hidden_dim)
        self.control_self_W = nn.Linear(in_features=control_dim, out_features=hidden_dim)

    def forward(self, x_int, x_bound, u, edge_index_int, edge_index_bound, edge_index_ctrl):
        x_int = x_int.squeeze(0)

        # computing the interior messages (from src: interior_row -> tgt: interior_col)
        # default flow is source_to_target, so messages belong to tgt and must be aggregated in tgt nodes
        interior_src, interior_tgt = edge_index_int
        interior_messages = torch.cat([x_int[interior_src], x_int[interior_tgt]], dim=1)
        interior_messages = self.message_int_int(interior_messages)
        # interior_messages = F.softplus(interior_messages)

        # computing the boundary messages
        # extracting subset of edge_index_bound that has boundary nodes as src nodes
        # default flow is source_to_target, so messages belong to boundary tgt and must be aggregated in tgt nodes
        edge_index_bound = edge_index_bound[:, torch.isin(edge_index_bound, self.boundary_node_index)[0]]
        boundary_src, boundary_tgt = edge_index_bound
        boundary_messages = torch.cat([x_bound, x_int[boundary_tgt]], dim=1)
        boundary_messages = self.message_bound_int(boundary_messages)
        # boundary_messages = F.softplus(boundary_messages)

        # computing the control messages
        # extracting subset of edge_index_ctrl that has ctrl nodes as src nodes
        # default flow is source_to_target, so messages belong to control tgt and must be aggregated in tgt nodes
        edge_index_ctrl = edge_index_ctrl[:, torch.isin(edge_index_ctrl, self.control_node_index)[0]]
        control_src, control_tgt = edge_index_ctrl
        control_messages = torch.cat([u, x_int[control_tgt]], dim=1)
        control_messages = self.message_ctrl_int(control_messages)
        # control_messages = F.softplus(control_messages)

        # aggregating the messages for the interior nodes
        target_nodes = torch.cat([interior_tgt, boundary_tgt, control_tgt], dim=0)
        target_messages = torch.cat([interior_messages, boundary_messages, control_messages], dim=0)
        agg_messages = self.message_aggregator(x=target_messages, index=target_nodes)

        # updating the interior nodes
        interior_message_update = self.interior_msg_W(agg_messages)
        interior_self_update = self.interior_self_W(x_int)
        interior_update = interior_self_update + interior_message_update

        # updating the boundary nodes
        self_boundary_messages = torch.cat([x_bound, x_bound], dim=1)
        self_boundary_messages = self.message_bound_bound(self_boundary_messages)
        # self_boundary_messages = F.softplus(self_boundary_messages)

        self_boundary_messages_updates = self.boundary_msg_W(self_boundary_messages)
        self_boundary_update = self.boundary_self_W(x_bound)
        boundary_update = self_boundary_update + self_boundary_messages_updates

        # updating the control nodes
        self_control_messages = torch.cat([u, u], dim=1)
        self_control_messages = self.message_ctrl_ctrl(self_control_messages)
        # self_control_messages = F.softplus(self_control_messages)

        self_control_messages_updates = self.control_msg_W(self_control_messages)
        self_control_update = self.control_self_W(u)
        control_update = self_control_update + self_control_messages_updates

        return interior_update, boundary_update, control_update


class DiffOp(nn.Module):
    def __init__(self, input_dim, control_dim, message_dim, hidden_dim,
                 interior_node_index, boundary_node_index, control_node_index):
        super(DiffOp, self).__init__()
        self.mp1 = BoundaryInjectedMessagePassingLayer(input_dim=input_dim, control_dim=control_dim,
                                                       message_dim=message_dim, hidden_dim=hidden_dim,
                                                       interior_node_index=interior_node_index,
                                                       boundary_node_index=boundary_node_index,
                                                       control_node_index=control_node_index)
        self.mp2 = BoundaryInjectedMessagePassingLayer(input_dim=hidden_dim, control_dim=hidden_dim,
                                                       message_dim=message_dim, hidden_dim=input_dim,
                                                       interior_node_index=interior_node_index,
                                                       boundary_node_index=boundary_node_index,
                                                       control_node_index=control_node_index)
        self.t_sensitivity = 0.0

    def diffop(self, x_int_t, x_bound_t, u_t, edge_index_int, edge_index_bound, edge_index_ctrl):
        # MP first layer
        dx_int_t_dt, x_bound_update, u_update = self.mp1(x_int_t, x_bound_t, u_t,
                                                         edge_index_int, edge_index_bound, edge_index_ctrl)
        # MP second layer
        dx_int_t_dt, _, _ = self.mp2(dx_int_t_dt, x_bound_update, u_update,
                                     edge_index_int, edge_index_bound, edge_index_ctrl)
        return dx_int_t_dt

    def forward(self, t, state):
        x_int_t, X_bound, U, edge_index_int, edge_index_bound, edge_index_ctrl, timestamps = state
        edge_index_int = edge_index_int.long()
        edge_index_bound = edge_index_bound.long()
        edge_index_ctrl = edge_index_ctrl.long()

        # interpolate X_bound and U and get X_bound(t) and U(t)
        x_bound_t, u_t = self._interpolation(t, timestamps, X_bound, U)

        # diffop
        dx_int_t_dt = self.diffop(x_int_t, x_bound_t, u_t, edge_index_int, edge_index_bound, edge_index_ctrl)

        # preparing state change
        dstate_dt = (
            dx_int_t_dt,
            torch.zeros_like(state[1]),
            torch.zeros_like(state[2]),
            torch.zeros_like(state[3]),
            torch.zeros_like(state[4]),
            torch.zeros_like(state[5]),
            torch.zeros_like(state[6]),
        )
        return dstate_dt

    @staticmethod
    def _interpolation(t, timestamps, X_bound, U):
        delta_t = 0.5
        if t > timestamps[-1]:
            return X_bound[-1], U[-1]

        if len(timestamps) == 1:
            indices = torch.zeros(1, 1, dtype=torch.long)
        #     k = timestamps[-1] / delta_t - 1

        else:
            indices = torch.where((timestamps[:-1] <= t) & (t <= timestamps[1:]))

        k = indices[0][-1].item()

        if k == timestamps.size(0) - 1:
            return X_bound[k], U[k]

        min_time = timestamps[k]
        max_time = timestamps[k + 1]
        # scaling t to 0<t<1
        scaled_t = (t - min_time) / (max_time - min_time)
        # linear interpolation of boundary_values_t
        X_boundary_t = (1 - scaled_t) * X_bound[k] + scaled_t * X_bound[k + 1]

        if k == len(U) - 1:
            U_t = (1 - scaled_t) * U[k]
        else:
            U_t = (1 - scaled_t) * U[k] + scaled_t * U[k + 1]

        return X_boundary_t, U_t


class BIGNODE(nn.Module):
    """
    SysID model for Dirichlet boundary conditions and controls.
    """
    def __init__(self, input_dim, control_dim, message_dim, hidden_dim,
                 interior_node_index, boundary_node_index, control_node_index):
        super(BIGNODE, self).__init__()
        self.diffop = DiffOp(input_dim=input_dim, control_dim=control_dim,
                             message_dim=message_dim, hidden_dim=hidden_dim,
                             interior_node_index=interior_node_index, boundary_node_index=boundary_node_index,
                             control_node_index=control_node_index)

    def forward(self, timestamps, x_int_0, X_bound, U, edge_index_int, edge_index_bound, edge_index_ctrl):
        # packing the state
        initial_state = (
            x_int_0,
            X_bound,
            U,
            edge_index_int,
            edge_index_bound,
            edge_index_ctrl,
            timestamps
        )

        X_int_hat = odeint(
            func=self.diffop,
            y0=initial_state,
            t=timestamps,
            method="dopri5",
        )[0]
        return X_int_hat
