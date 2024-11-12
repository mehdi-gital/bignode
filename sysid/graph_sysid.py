import os
import random
import pickle

import numpy as np
import torch

from bignode_no_ctrl import BIGNODE
from utils_sysid import train, inference, compute_RMSE
from utils_visualization import visualize_state_sysid
from utils import experiment_init


random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(23)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Change this as needed:
    SYSTEM_TYPE = "nonlinear"
    REGULARIZATION = True
    if SYSTEM_TYPE == "linear":
        DATA_DIR = "data/linear"
        MESSAGE_DIM = 32
        EMBED_DIM = 32
        EXPERIMENT_NAME = "bignode_reg_linear" if REGULARIZATION else "bignode_linear"
    elif SYSTEM_TYPE == "nonlinear":
        DATA_DIR = "data/nonlinear"
        MESSAGE_DIM = 64
        EMBED_DIM = 64
        EXPERIMENT_NAME = "bignode_reg_nonlinear" if REGULARIZATION else "bignode_nonlinear"
    else:
        raise NotImplementedError
    EPOCHS = 20

    # Load graph data
    with open(os.path.join(DATA_DIR, "graph_obj.pickle"), "rb") as handle:
        graph_obj = pickle.load(handle)
    num_nodes_int = graph_obj["num_nodes_int"]
    num_nodes_bound = graph_obj["num_nodes_bound"]

    # Load system data
    with open(os.path.join(DATA_DIR, "system_data.pickle"), "rb") as handle:
        system_data = pickle.load(handle)

    # Initializing the experiment
    experiment_dirs = experiment_init(experiment_name=EXPERIMENT_NAME, graph_obj=graph_obj, system_data=system_data)

    # Initializing the model
    bignode = BIGNODE(input_dim=1, message_dim=MESSAGE_DIM, embed_dim=EMBED_DIM, bvp_type="dirichlet",
                      num_nodes_int=num_nodes_int, num_nodes_bound=num_nodes_bound, device=device)

    # Training
    model, losses = train(graph_obj=graph_obj,
                  system_data=system_data,
                  regularization=REGULARIZATION,
                  model=bignode,
                  model_dir=experiment_dirs["model_dir"],
                  epochs=EPOCHS)

    # Training evaluation
    X_int_hat = inference(graph_obj, system_data, model)
    visualize_state_sysid(system_type=SYSTEM_TYPE,
                          graph_obj=graph_obj,
                          X_bound=system_data["X_bound"],
                          X_int=system_data["X_int"],
                          X_int_hat=X_int_hat,
                          visualization_dir=experiment_dirs["visualization_dir"])
    RMSE = compute_RMSE(
        X_int_true=system_data["X_int"],
        X_int_pred=X_int_hat,
    )
    print(f'[*] RMSE: {RMSE}')
    print(experiment_dirs)


if __name__ == "__main__":
    main()

