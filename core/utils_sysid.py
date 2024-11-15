import os
import time
from typing import Dict, Union, Any, Tuple, List

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index, to_networkx

# from utils_data import generate_rand_graph, generate_diffusion_data


def distance2bound(graph_obj: Dict[str, Union[int, torch.Tensor]]):
    """
    Returns a subgraph of size at most subgraph_size consisting of nodes that are likely to be far from the source of heat
    """
    # Need to point to the right directory for plot saving
    num_nodes_int = graph_obj["num_nodes_int"]
    num_nodes_bound = graph_obj["num_nodes_bound"]
    num_nodes = num_nodes_int + num_nodes_bound
    edge_index_int = graph_obj["edge_index_int"]
    edge_index_bound = graph_obj["edge_index_bound"]
    edge_index = sort_edge_index(torch.cat([edge_index_int, edge_index_bound], dim=1))
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    nx_graph = to_networkx(data, to_undirected=True)

    paths = dict(nx.all_pairs_shortest_path(nx_graph))

    # find the distance between any interior node and the boundary
    distance2bound = {}

    for i in range(graph_obj['num_nodes_int']):
        distances = []
        for j in range(graph_obj['num_nodes_int'], graph_obj['num_nodes_int'] + graph_obj['num_nodes_bound']):
            d = paths[i][j]
            distances.append(len(d))
        distance2bound[i] = min(distances) - 1

    distance_counts = list(distance2bound.values())
    # selected_indices = random.sample(range(graph_obj['num_nodes_int']), k=subgraph_size, counts=distance_counts)

    return distance_counts


def custom_mse_loss(input_tensor, target_tensor, coordinate_weights=None, reduction='mean'):
    """
    Custom MSE loss with per-coordinate weights.

    Args:
    - input (Tensor): The input tensor.
    - target (Tensor): The target tensor.
    - coordinate_weights (Tensor): A tensor of per-coordinate weights. Should have the same shape as input and target.
    - reduction (str, optional): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').

    Returns:
    - Tensor: The computed loss.
    """

    # Compute squared differences
    squared_diff = (input_tensor - target_tensor) ** 2

    # Apply per-coordinate weights if provided
    if coordinate_weights is not None:
        squared_diff = squared_diff * coordinate_weights

    # Compute the mean or sum based on the reduction parameter
    if reduction == 'mean':
        return torch.mean(squared_diff)
    elif reduction == 'sum':
        return torch.sum(squared_diff)
    elif reduction == 'none':
        return squared_diff
    else:
        raise ValueError("Invalid reduction. Use 'mean', 'sum', or 'none'.")


def train(
        graph_obj: Dict[str, Union[int, torch.Tensor]],
        system_data: Dict[str, Union[str, torch.Tensor]],
        regularization: bool,
        model: nn.Module,
        model_dir: str,
        epochs: int
) -> Tuple[nn.Module, List[float]]:
    edge_index_int = graph_obj["edge_index_int"]
    edge_index_bound = graph_obj["edge_index_bound"]
    edge_index_ctrl = graph_obj["edge_index_ctrl"]
    has_ctrl = True if edge_index_ctrl is not None else False
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=150,
        factor=0.8
    )

    prev_lr = optimizer.param_groups[0]["lr"]

    losses = []

    model.train()
    best_epoch = 0
    min_loss = 1e40
    for epoch in range(epochs):
        s = time.time()
        # get batch data
        batch_x_int_0, batch_t, batch_X_int, batch_X_bound, batch_U, batches = get_batch(
            dataset=system_data,
            batch_size=32,
            batch_time=50,
        )

        # forward pass
        if has_ctrl:
            X_int_hat = model(batch_t, batch_x_int_0, batch_X_bound, batch_U, edge_index_int, edge_index_bound,
                              edge_index_ctrl)
        else:
            X_int_hat = model(batch_t, batch_x_int_0, batch_X_bound, edge_index_int, edge_index_bound)

        # loss computation
        if regularization:
            dists = torch.tensor(distance2bound(graph_obj))
            loss_weights = torch.exp(dists)
            loss_weights[torch.where(dists == 1)[0]] = 1.0
            loss = custom_mse_loss(X_int_hat, batch_X_int, loss_weights)
        else:
            loss = F.mse_loss(X_int_hat, batch_X_int)

        losses.append(loss.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print(f"====== epoch: {epoch}, Learning Rate changed to: {current_lr}")
            prev_lr = current_lr

        if epoch < 10 or epoch % 50 == 0:
            print('epoch: {}, loss: {:.4f}, time: {:.1f}'.format(epoch, loss.item(), time.time() - s))

        if loss.item() <= min_loss:
            min_loss = loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, f'bignode.pth'))
            if epoch % 10 == 0:
                print(f'--- best model saved at epoch {epoch} with loss {loss.item()} ---')
    print(f'[*] Training completed. Best model saved at epoch {best_epoch} with loss {min_loss}')

    # eval
    print('[*] loading the best model and setting it to eval mode.')
    model.load_state_dict(torch.load(os.path.join(model_dir, f'bignode.pth')))
    model.eval()
    return model, losses


def inference(
        graph_obj: Dict[str, Union[int, torch.Tensor]],
        system_data: Dict[str, Union[str, torch.Tensor]],
        model: nn.Module,
) -> torch.Tensor:
    edge_index_int = graph_obj["edge_index_int"]
    edge_index_bound = graph_obj["edge_index_bound"]
    edge_index_ctrl = graph_obj["edge_index_ctrl"]
    has_ctrl = True if edge_index_ctrl is not None else False
    X_int = system_data["X_int"]
    X_bound = system_data["X_bound"]
    U = system_data["U"]
    timestamps = system_data["timestamps"]
    with torch.no_grad():
        if has_ctrl:
            X_int_hat = model(timestamps, X_int[0].unsqueeze(0), X_bound, U, edge_index_int,
                              edge_index_bound, edge_index_ctrl)
        else:
            X_int_hat = model(timestamps, X_int[0].unsqueeze(0), X_bound, edge_index_int, edge_index_bound)
    X_int_hat = X_int_hat.squeeze().unsqueeze(-1)
    return X_int_hat


def get_batch(
    dataset: Dict,
    batch_size: int,
    batch_time: int,
):
    """returns a random subsegment of the timed data and observes the initial value within the segment."""
    X_int = dataset["X_int"]
    t = dataset["timestamps"]
    X_bound = dataset["X_bound"]
    U = dataset["U"]
    num_timesteps = len(t)
    has_ctrl = True if U is not None else False
    batches = torch.from_numpy(
        np.random.choice(
            np.arange(num_timesteps - batch_time, dtype=np.int64),
            batch_size,
            replace=False,
        )
    )  # (batch_size, )
    batch_t = t[:batch_time]  # (batch_time, )
    batch_x_int_0 = X_int[batches, :, :].float()  # (batch_size, N, D)
    batch_X_int = []
    batch_X_bound = []
    batch_U = []
    for batch in batches:
        xi = []
        xb = []
        u = []
        for t in range(batch_time):
            xi.append(X_int[batch + t].unsqueeze(0))
            xb.append(X_bound[batch + t].unsqueeze(0))
            if has_ctrl:
                u.append(U[batch + t].unsqueeze(0))
        xi = torch.cat(xi, dim=0).unsqueeze(1)
        xb = torch.cat(xb, dim=0).unsqueeze(1)
        if has_ctrl:
            u = torch.cat(u, dim=0).unsqueeze(1)
        batch_X_int.append(xi)
        batch_X_bound.append(xb)
        batch_U.append(u)
    batch_X_int = torch.cat(batch_X_int, dim=1)  # (batch_time, batch_size, N, D)
    batch_X_bound = torch.cat(batch_X_bound, dim=1)
    if has_ctrl:
        batch_U = torch.cat(batch_U, dim=1).float()
    return (
        batch_x_int_0.float(),
        batch_t.float(),
        batch_X_int.float(),
        batch_X_bound.float(),
        batch_U,
        batches,
    )


def generate_data(
        graph_type: str,
        graph_params: Dict[str, Union[float]],
        num_nodes_int: int,
        num_nodes_bound: int,
        dt: float,
        diffusion_coeff: float,
        bvp_type: str,
        x_int_0: torch.Tensor = None,
        device: Any = torch.device("cpu")
) -> Tuple[Dict[str, Union[int, torch.Tensor]], Dict[str, Union[str, torch.Tensor]]]:
    num_nodes_ctrl = None
    graph_obj = generate_rand_graph(graph_type, graph_params, num_nodes_int, num_nodes_bound, device, num_nodes_ctrl)

    # generate data on the graph
    N = int(1.0 / dt)
    if not x_int_0:
        x_int_0 = torch.zeros(num_nodes_int, 1, dtype=torch.float)
    system_data = generate_diffusion_data(graph_obj, x_int_0, device, bvp_type, N, dt, diffusion_coeff)
    return graph_obj, system_data


def compute_RMSE(
        X_int_true: torch.Tensor,
        X_int_pred: torch.Tensor
):
    return torch.sqrt(F.mse_loss(input=X_int_pred, target=X_int_true)).item()


def train_test_split(
        system_data: Dict[str, Union[str, torch.Tensor]],
        train_test_ratio: float

) -> Tuple[Dict[str, Union[str, torch.Tensor]], Dict[str, Union[str, torch.Tensor]]]:
    num_timesteps = system_data["X_int"].size(0)
    train_end_timestep = int(train_test_ratio * num_timesteps)
    if system_data["U"]:
        U_train = system_data["U"][:train_end_timestep, :, :]
        U_test = system_data["U"][train_end_timestep:, :, :]
    else:
        U_train = system_data["U"]
        U_test = system_data["U"]
    train_system_data = {
        "system_name": system_data["system_name"],
        "X_int": system_data["X_int"][:train_end_timestep, :, :],
        "X_bound": system_data["X_bound"][:train_end_timestep, :, :],
        "U": U_train,
        "timestamps": system_data["timestamps"][:train_end_timestep],
    }
    test_system_data = {
        "system_name": system_data["system_name"],
        "X_int": system_data["X_int"][train_end_timestep:, :, :],
        "X_bound": system_data["X_bound"][train_end_timestep:, :, :],
        "U": U_test,
        "timestamps": system_data["timestamps"][train_end_timestep:],
    }
    return train_system_data, test_system_data
