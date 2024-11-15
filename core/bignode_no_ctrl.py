from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torchdiffeq import odeint


class BoundaryInjectedMessagePassingLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            message_dim: int,
            embed_dim: int,
            num_nodes_int: int,
            num_nodes_bound: int,
            device: Any
    ):
        super(BoundaryInjectedMessagePassingLayer, self).__init__()
        self.node_index_int = torch.arange(0, num_nodes_int, dtype=torch.long).to(device)
        self.node_index_bound = torch.arange(num_nodes_int, num_nodes_int + num_nodes_bound,
                                             dtype=torch.long).to(device)

        self.message_function = nn.Linear(in_features=2 * input_dim, out_features=message_dim)
        self.message_aggregator = aggr.MeanAggregation()
        self.update_self = nn.Linear(in_features=input_dim, out_features=embed_dim)
        self.update_message = nn.Linear(in_features=message_dim, out_features=embed_dim)

    def forward(self, x_int, bv, edge_index_int, edge_index_bound):
        # x_int: [N, Di]
        # edge_index_int: [2, M], src_int, dst_int: [M]
        # x_int[src_int], x_int[dst_int]: [M, Di]
        # msg_int: [M, Dm]
        if len(x_int.size()) == 1 or len(x_int.size()) == 2:
            x_int = x_int.reshape(1, -1, 1)
        src_int, dst_int = edge_index_int
        msg_int = self.message_function(torch.cat([x_int[:, src_int], x_int[:, dst_int]], dim=-1))

        edge_index_bound = edge_index_bound[:, torch.isin(edge_index_bound, self.node_index_bound)[0]]
        src_bound, dst_bound = edge_index_bound
        # msg_bound: [M', Dm] --> bv[:, src_bound-len(self.node_index_int)] before: bv
        msg_bound = self.message_function(torch.cat([bv[:, src_bound-len(self.node_index_int)],
                                                     x_int[:, dst_bound]], dim=-1))

        all_msg = torch.cat([msg_int, msg_bound], dim=1)
        dst_nodes = torch.cat([dst_int, dst_bound], dim=0)
        # agg_msg_int: [N, Dm]
        agg_msg_int = self.message_aggregator(all_msg, dst_nodes)

        # updating
        x_int_update = self.update_self(x_int) + self.update_message(agg_msg_int)
        bv_update = self.update_self(bv)

        return x_int_update, bv_update


class DiffOp(nn.Module):
    """Differential Operator"""
    def __init__(
            self,
            input_dim: int,
            message_dim: int,
            embed_dim: int,
            bvp_type: str,
            num_nodes_int: int,
            num_nodes_bound: int,
            device: Any
    ):
        super(DiffOp, self).__init__()
        self.bvp_type = bvp_type
        self.num_nodes_int = num_nodes_int
        self.num_nodes_bound = num_nodes_bound
        self.node_index_int = torch.arange(0, num_nodes_int, dtype=torch.long)
        self.node_index_bound = torch.arange(num_nodes_int, num_nodes_int + num_nodes_bound, dtype=torch.long)

        self.mp1 = BoundaryInjectedMessagePassingLayer(input_dim, message_dim, embed_dim,
                                                       num_nodes_int, num_nodes_bound, device)
        self.mp2 = BoundaryInjectedMessagePassingLayer(embed_dim, message_dim, embed_dim,
                                                       num_nodes_int, num_nodes_bound, device)
        self.mp3 = BoundaryInjectedMessagePassingLayer(embed_dim, message_dim, embed_dim,
                                                       num_nodes_int, num_nodes_bound, device)
        self.mp4 = BoundaryInjectedMessagePassingLayer(embed_dim, message_dim, input_dim,
                                                       num_nodes_int, num_nodes_bound, device)

    def message_passing_func(self, x_int_t, bv_t, edge_index_int, edge_index_bound):

        if len(bv_t.size()) == 2:
            bv_t = bv_t.unsqueeze(0)
        # mp1
        x_int_update, bv_update = self.mp1(x_int_t, bv_t, edge_index_int, edge_index_bound)
        # x_int_update = F.relu(x_int_update)
        # bv_update = F.relu(bv_update)
        x_int_update = F.softplus(x_int_update)
        bv_update = F.softplus(bv_update)

        # mp2
        x_int_update, bv_update = self.mp2(x_int_update, bv_update, edge_index_int, edge_index_bound)
        x_int_update = F.softplus(x_int_update)
        bv_update = F.softplus(bv_update)

        # mp3
        x_int_update, bv_update = self.mp3(x_int_update, bv_update, edge_index_int, edge_index_bound)
        x_int_update = F.softplus(x_int_update)
        bv_update = F.softplus(bv_update)

        # mp4
        x_int_update, bv_update = self.mp4(x_int_update, bv_update, edge_index_int, edge_index_bound)

        return x_int_update, bv_update

    def forward(self, t, state):
        x_int_t, boundary_values, edge_index_int, edge_index_bound, timestamps = state
        edge_index_int = edge_index_int.long()
        edge_index_bound = edge_index_bound.long()

        # get bv_t from boundary_values
        bv_t = self.piecewise_lines(t, timestamps, boundary_values)
        if len(bv_t.size()) == 2:
            bv_t = bv_t.unsqueeze(0)

        if self.bvp_type == "neumann":
            bound_nodes, int_nodes = edge_index_bound[:, torch.isin(edge_index_bound, self.node_index_bound)[0]]
            bound_nodes = bound_nodes - self.num_nodes_int
            bv_t = bv_t[:, bound_nodes, :] + x_int_t[:, int_nodes, :]

        x_int_update, bv_update = self.message_passing_func(x_int_t, bv_t, edge_index_int, edge_index_bound)

        dstate_dt = (
            x_int_update,
            torch.zeros_like(state[1]),
            torch.zeros_like(state[2]),
            torch.zeros_like(state[3]),
            torch.zeros_like(state[4])
        )
        return dstate_dt

    @staticmethod
    def piecewise_lines(t, timestamps, boundary_values):
        # Find the indices of the two closest timestamps
        idx = torch.searchsorted(timestamps, t)
        idx_left = max(idx - 1, 0)
        idx_right = min(idx, len(timestamps) - 1)

        # Corresponding timestamps and values
        t_left, t_right = timestamps[idx_left], timestamps[idx_right]
        try:
            value_left, value_right = boundary_values[idx_left], boundary_values[idx_right]
        except Exception as e:
            pass

        # Perform linear interpolation
        if t_left == t_right:
            # If the two timestamps are the same, return the corresponding value
            return value_left
        else:
            # Linear interpolation formula
            interpolated_value = value_left + (t - t_left) * (value_right - value_left) / (t_right - t_left)
            return interpolated_value


class BIGNODE(nn.Module):
    """Boundary Injected Graph Neural ODE"""

    def __init__(
            self,
            input_dim: int,
            message_dim: int,
            embed_dim: int,
            bvp_type: str,
            num_nodes_int: int,
            num_nodes_bound: int,
            device: Any
    ):
        super(BIGNODE, self).__init__()
        assert bvp_type in ["dirichlet", "neumann"]
        self.bvp_type = bvp_type
        self.diffop = DiffOp(input_dim, message_dim, embed_dim, bvp_type, num_nodes_int, num_nodes_bound, device)

    def forward(
            self,
            timestamps: torch.Tensor,
            x_int_0: torch.Tensor,
            bound_values: torch.Tensor,
            edge_index_int: torch.Tensor,
            edge_index_bound: torch.Tensor,
    ):
        initial_state = (x_int_0, bound_values, edge_index_int, edge_index_bound, timestamps)
        X_int_hat = odeint(
            func=self.diffop,
            y0=initial_state,
            t=timestamps,
            method="dopri5",
        )[0]
        return X_int_hat
