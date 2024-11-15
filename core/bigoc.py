import os
import pickle
import time
import random
from typing import Callable, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(23)


class BIGOC:
    """
    Boundary Injected Graph Neural ODE Optimal Control
    """
    def __init__(
            self,
            diffop: nn.Module,
            N: int,
            dt: float,
            stage_cost: Callable,
            terminal_cost: Callable,
            graph_obj: Dict[str, Union[int, torch.Tensor]],
            ocp_obj: Dict[str, Union[float, int, torch.Tensor]],
            checkpoint_dir: str
    ):
        self.checkpoint_dir = checkpoint_dir
        self.diffop = diffop
        self.dt = dt
        self.N = N

        # cost functions
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost

        # graph and ocp objects
        self.graph_obj = graph_obj
        self.ocp_obj = ocp_obj

        self.timestamps = torch.tensor([i * self.dt for i in range(self.N + 1)])

    def f(self, x_int_k, X_bound, U, t_k):
        # Runge–Kutta Method (explicit, K = 4)
        state_k_1 = (x_int_k,
                     X_bound, U,
                     self.graph_obj["edge_index_int"], self.graph_obj["edge_index_bound"],
                     self.graph_obj["edge_index_ctrl"], self.timestamps)
        t_k_1 = t_k
        k_1 = self.dt * self.diffop(t_k_1, state_k_1)[0]

        # k_2
        state_k_2 = (x_int_k + 0.5 * k_1,
                     X_bound, U,
                     self.graph_obj["edge_index_int"], self.graph_obj["edge_index_bound"],
                     self.graph_obj["edge_index_ctrl"], self.timestamps)
        t_k_2 = t_k + self.dt / 2
        k_2 = self.dt * self.diffop(t_k_2, state_k_2)[0]

        # k_3
        state_k_3 = (x_int_k + 0.5 * k_2,
                     X_bound, U,
                     self.graph_obj["edge_index_int"], self.graph_obj["edge_index_bound"],
                     self.graph_obj["edge_index_ctrl"], self.timestamps)
        t_k_3 = t_k + self.dt / 2
        k_3 = self.dt * self.diffop(t_k_3, state_k_3)[0]

        # k_4
        state_k_4 = (x_int_k + k_3,
                     X_bound, U,
                     self.graph_obj["edge_index_int"], self.graph_obj["edge_index_bound"],
                     self.graph_obj["edge_index_ctrl"], self.timestamps)
        t_k_4 = t_k + self.dt
        k_4 = self.dt * self.diffop(t_k_4, state_k_4)[0]

        # x_{i+1}
        x_int_k_plus_one = x_int_k + (1.0/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

        return x_int_k_plus_one

    def L(self, x_k, u_k, x_k_plus_one, u_k_plus_one, x_int_ref):
        # trapezoid approximation
        cost_k = self.stage_cost(x_k, u_k, x_int_ref,
                                 self.ocp_obj['state_coeff'], self.ocp_obj['control_coeff'])
        cost_k_plus_one = self.stage_cost(x_k_plus_one, u_k_plus_one, x_int_ref,
                                          self.ocp_obj['state_coeff'], self.ocp_obj['control_coeff'])
        L_k = (self.dt / 2) * (cost_k + cost_k_plus_one)
        return L_k

    def solve(
            self,
            x_int_0: torch.Tensor,
            x_int_des: torch.Tensor,
            X_bound: torch.Tensor,
            method: str = "BFGS",
            epsilon: float = 1e-6,
            max_iter: int = 100,
    ):
        # initial U
        U = torch.rand(self.N, self.graph_obj["num_nodes_ctrl"], 1)

        # simulation
        X_int, Lambda = self.simulation(
            x_int_0=x_int_0,
            X_bound=X_bound,
            U=U,
            x_int_des=x_int_des,
        )
        KKT = self.grad_U_Lagrangian(X_int, Lambda, U, X_bound, x_int_des, method)
        KKT_norm = torch.linalg.norm(KKT)
        best_norm = 1e10

        if method == "BFGS":
            H_k = torch.eye(self.N * self.graph_obj["num_nodes_ctrl"])
        elif method == "Newton":
            H_k = None
        else:
            raise NotImplementedError

        itr = 0
        while KKT_norm > epsilon and itr < max_iter:
            iter_start_time = time.time()
            if method == "BFGS":
                direction_i = - H_k @ KKT.reshape(-1, 1)
                direction_i = direction_i.reshape(self.N, self.graph_obj["num_nodes_ctrl"], 1)
                U_k = U.detach().clone()
                KKT_k = KKT.detach().clone()
                step_length_i = 1
            elif method == "Newton":
                jacob_KKT = jacrev(self.grad_U_Lagrangian, argnums=2)(X_int, Lambda,
                                                                      U, X_bound, x_int_des, method).squeeze()
                jacob_KKT = jacob_KKT.detach()
                jacob_KKT_inv = torch.linalg.inv(jacob_KKT)
                direction_i = - jacob_KKT_inv @ KKT.squeeze(-1)
                del KKT
                del jacob_KKT
                direction_i = direction_i.unsqueeze(-1)
                step_length_i = 1e-3
                U_k = None
                KKT_k = None
            else:
                raise NotImplementedError

            # update
            U.add_(step_length_i * direction_i)

            # simulation
            X_int, Lambda = self.simulation(
                x_int_0=x_int_0,
                X_bound=X_bound,
                U=U,
                x_int_des=x_int_des,
            )
            KKT = self.grad_U_Lagrangian(X_int, Lambda, U, X_bound, x_int_des, method)
            KKT_norm = torch.linalg.norm(KKT)

            # bfgs_start_time = time.time()
            if method == "BFGS":
                S_k = (U - U_k).reshape(-1, 1)
                Y_k = (KKT - KKT_k).reshape(-1, 1)
                H_k = H_k + ((S_k.T @ Y_k + Y_k.T @ H_k @ Y_k) * (S_k @ S_k.T)) / (S_k.T @ Y_k)**2 - (H_k @ Y_k @ S_k.T + S_k @ Y_k.T @ H_k)/(S_k.T @ Y_k)
            itr += 1

            iter_end_time = time.time()
            print('[*] iter: {}, KKT_norm: {}, step_length: {}, time: {:.1f}'.format(itr, KKT_norm, step_length_i,
                                                                                     iter_end_time - iter_start_time))

            # saving
            if KKT_norm < best_norm:
                best_norm = KKT_norm
                print('--saving best U')
                if method == "BFGS":
                    data = {
                        "N": self.N,
                        "dt": self.dt,
                        "ocp_obj": self.ocp_obj,
                        "U": U,
                        "x_int_des": x_int_des,
                        "x_int_0": X_int[0],
                        "X_int": X_int,
                        "X_bound": X_bound,
                        "itr": itr,
                        "H_k": H_k,
                        "U_guess": "rand",
                        "H_k_init": "eye",
                        "best_KKT_norm": best_norm,
                    }
                else:
                    data = {
                        "U": U,
                        "x_int_des": x_int_des,
                        "X_bound": X_bound,
                        "itr": itr
                    }
                with open(os.path.join(self.checkpoint_dir, f"bigoc_{method}.pickle"), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.checkpoint_dir, f"bigoc_{method}.pickle"), 'rb') as handle:
            data = pickle.load(handle)
        U = data['U']
        # forward sweep
        X_int = [x_int_0.unsqueeze(0).detach()]
        for k in range(self.N):
            x_int_k = X_int[k]
            t_k = self.timestamps[k]
            x_int_next = self.f(x_int_k, X_bound, U, t_k)
            X_int.append(x_int_next.detach())
        X_int = torch.cat(X_int, dim=0)
        U = U.squeeze(-1).detach().numpy()
        X_int = X_int.squeeze(-1).detach().numpy()
        return U, X_int

    def simulation(
            self,
            x_int_0: torch.Tensor,
            X_bound: torch.Tensor,
            U: torch.Tensor,
            x_int_des: torch.Tensor,


    ):
        # forward sweep
        X_int = [x_int_0.unsqueeze(0).detach()]
        for k in range(self.N):
            x_int_k = X_int[k]
            t_k = self.timestamps[k]
            x_int_next = self.f(x_int_k, X_bound, U, t_k)
            X_int.append(x_int_next.detach())
        X_int = torch.cat(X_int, dim=0)

        # backward sweep
        x_int_N = X_int[-1]
        lambda_N = jacrev(self.terminal_cost, argnums=0)(x_int_N, x_int_des, self.ocp_obj['terminal_cost_coeff'])
        Lambda_reversed = [lambda_N.detach()]
        for k in range(self.N - 1, 0, -1):  # k=N-1, ..., 1
            x_int_k = X_int[k]
            u_k = U[k - 1]
            t_k = self.timestamps[k]
            x_int_k_plus_one = X_int[k + 1]
            u_k_plus_one = U[k]
            grad_x_k_L = jacrev(self.L, argnums=0)(x_int_k, u_k, x_int_k_plus_one, u_k_plus_one, x_int_des)
            jacob_x_k_f = jacrev(self.f, argnums=0)(x_int_k.unsqueeze(0), X_bound, U, t_k).squeeze()
            lambda_k_plus_one = Lambda_reversed[-1]  # k+1=N, ...., 2
            lambda_k = grad_x_k_L + jacob_x_k_f.T @ lambda_k_plus_one
            Lambda_reversed.append(lambda_k.detach())
        Lambda_reversed.append(torch.zeros_like(Lambda_reversed[0]))
        Lambda = torch.cat([lambda_k.unsqueeze(0) for lambda_k in Lambda_reversed[::-1]], dim=0)
        return X_int, Lambda

    def grad_U_Lagrangian(
            self,
            X_int: torch.Tensor,
            Lambda: torch.Tensor,
            U: torch.Tensor,
            X_bound: torch.Tensor,
            x_int_des: torch.Tensor,
            method: str
    ):
        kkt = []
        for k in range(self.N):  # k=0, ...., N-1
            x_int_k = X_int[k]
            u_k = U[k]
            t_k = self.timestamps[k]
            lambda_k_plus_one = Lambda[k + 1]
            x_int_k_plus_one = X_int[k + 1]
            if k == self.N - 1:
                u_k_plus_one = U[k]
            else:
                u_k_plus_one = U[k + 1]
            grad_u_k_L = jacrev(self.L, argnums=1)(x_int_k, u_k, x_int_k_plus_one, u_k_plus_one, x_int_des)
            jacob_u_k_f = jacrev(self.f, argnums=2)(x_int_k.unsqueeze(0), X_bound, U, t_k).squeeze()[:, k].unsqueeze(-1)
            partial_kkt = grad_u_k_L + jacob_u_k_f.T @ lambda_k_plus_one
            if method == "BFGS":
                kkt.append(partial_kkt.unsqueeze(0).detach())
            elif method == "Newton":
                kkt.append(partial_kkt.unsqueeze(0))
            else:
                raise NotImplementedError
        kkt = torch.cat(kkt, dim=0)
        return kkt
