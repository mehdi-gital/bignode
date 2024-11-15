import sys
import os
import random
import pickle
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from core.bignode_with_ctrl import BIGNODE
from core.bigoc import BIGOC
from core.utils_visualization import visualize_state_ocp, visualize_ctrl
from core.utils import experiment_init


random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(23)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stage_cost(
        x_int_k: torch.Tensor,
        u_k: torch.Tensor,
        x_int_ref: torch.Tensor,
        state_coeff: torch.Tensor,
        control_coeff: torch.Tensor,

) -> torch.Tensor:
    state_cost = state_coeff * torch.linalg.norm(x_int_k - x_int_ref) ** 2
    control_cost = control_coeff * torch.linalg.norm(u_k) ** 2
    cost = state_cost + control_cost
    return cost


def terminal_cost(
        x_int_N: torch.Tensor,
        x_int_ref: torch.Tensor,
        terminal_cost_coeff: torch.Tensor
) -> torch.Tensor:
    cost = terminal_cost_coeff * torch.linalg.norm(x_int_N - x_int_ref) ** 2
    return cost


def main():
    METHOD = "BFGS"
    DATA_DIR = os.path.join("data", "control")

    # loading graph_obj
    with open(os.path.join(DATA_DIR, "graph_obj_0.pickle"), "rb") as handle:
        graph_obj = pickle.load(handle)
    num_nodes_int = graph_obj["num_nodes_int"]
    num_nodes_bound = graph_obj["num_nodes_bound"]
    num_nodes_ctrl = graph_obj["num_nodes_ctrl"]
    V_int = torch.arange(0, num_nodes_int, dtype=torch.long)
    V_bound = torch.arange(num_nodes_int, num_nodes_int + num_nodes_bound, dtype=torch.long)
    V_ctrl = torch.arange(num_nodes_int + num_nodes_bound,
                          num_nodes_int + num_nodes_bound + num_nodes_ctrl, dtype=torch.long)

    # loading the model
    bignode = BIGNODE(input_dim=1,
                      control_dim=1,
                      message_dim=16,
                      hidden_dim=32,
                      interior_node_index=V_int,
                      boundary_node_index=V_bound,
                      control_node_index=V_ctrl)
    CHECKPOINT_PATH = os.path.join(DATA_DIR, "bignode.pth")
    bignode.load_state_dict(torch.load(CHECKPOINT_PATH))
    bignode.eval()

    # initializing the experiment
    EXPERIMENT_NAME = f"bigoc_{METHOD}"
    experiment_dirs = experiment_init(experiment_name=EXPERIMENT_NAME, graph_obj=graph_obj)

    # OCP problem
    N = 50
    dt = 0.1
    ocp_obj = {
        "state_coeff": 50.0,
        "control_coeff": 1.0,
        "terminal_cost_coeff": 10_000.0,
        "N": N,
        "dt": 0.1
    }

    # bignode dynamics + ctrl
    x_int_0 = torch.zeros(num_nodes_int, 1, dtype=torch.float)
    x_int_des = torch.tensor([16, 12, 6], dtype=torch.float).reshape(-1, 1)
    X_bound = torch.zeros(N + 1, 2, 1)

    # initializing the optimizer
    optimizer = BIGOC(
        diffop=bignode.diffop,
        N=N,
        dt=dt,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        graph_obj=graph_obj,
        ocp_obj=ocp_obj,
        checkpoint_dir=experiment_dirs["data_dir"]
    )
    U, X_int = optimizer.solve(x_int_0=x_int_0,
                               x_int_des=x_int_des,
                               X_bound=X_bound,
                               method=METHOD,
                               max_iter=100)
    visualize_state_ocp(X_int_bigoc=X_int,
                        timestamps=optimizer.timestamps,
                        x_int_des=x_int_des,
                        num_nodes_int=num_nodes_int,
                        visualization_dir=experiment_dirs["visualization_dir"])
    visualize_ctrl(U_bigoc=U,
                   timestamps=optimizer.timestamps,
                   num_nodes_int=num_nodes_int,
                   num_nodes_bound=num_nodes_bound,
                   num_nodes_ctrl=num_nodes_ctrl,
                   visualization_dir=experiment_dirs["visualization_dir"])


if __name__ == "__main__":
    main()
