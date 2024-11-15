import os

import torch

import matplotlib.pyplot as plt


def visualize_state_sysid(system_type: str,
                          graph_obj,
                          X_bound,
                          X_int,
                          X_int_hat,
                          visualization_dir=None):
    num_nodes_int = graph_obj["num_nodes_int"]
    edge_index_bound = graph_obj["edge_index_bound"]

    timestamps = [i for i in range(X_int.shape[0])]
    for i in range(num_nodes_int):
        boundary_neighbors_idx = torch.where(edge_index_bound[0] == i)[0]
        X_int_i = X_int[:, i]
        X_int_hat_i = X_int_hat[:, i]

        if len(boundary_neighbors_idx) == 0:
            fig, axes = plt.subplots(1, 1, figsize=(24, 10))
            axes = [axes]

            axes[0].set_title(f"Interior node {i}", fontsize=48)
            axes[0].plot(timestamps, X_int_i, label=f'{system_type} Diffusion', c='green', linewidth=8, linestyle='--')
            axes[0].plot(timestamps, X_int_hat_i, linewidth=4, label='BigNode', c='blue')
            axes[0].set_facecolor("whitesmoke")
            axes[0].grid(True, color='gray', linewidth=0.5)  # Add grid lines
            axes[0].set_xlabel("Time Step k", fontsize=32)
            axes[0].set_ylabel("State", fontsize=32)
            axes[0].tick_params(axis='both', which='major', labelsize=24)
            axes[0].legend()
            axes[0].legend(loc='upper left', fontsize=32)

            plt.tight_layout()
            if visualization_dir:
                plt.savefig(os.path.join(visualization_dir, f"state_{i}.png"), dpi=150)
            else:
                plt.show()
            plt.clf()
            plt.close()
        else:
            fig, axes = plt.subplots(len(boundary_neighbors_idx) + 1, 1, figsize=(24, 10))

            axes[0].set_title(f"Interior node {i}", fontsize=48)
            axes[0].plot(timestamps, X_int_i, label=f'{system_type} Diffusion', c='green', linewidth=8, linestyle='--')
            axes[0].plot(timestamps, X_int_hat_i, linewidth=4, label='BigNode', c='blue')
            axes[0].set_facecolor("whitesmoke")
            axes[0].grid(True, color='gray', linewidth=0.5)  # Add grid lines
            axes[0].set_xlabel("Time Step k", fontsize=32)
            axes[0].set_ylabel("State", fontsize=32)
            axes[0].tick_params(axis='both', which='major', labelsize=24)
            axes[0].legend()
            axes[0].legend(loc='upper left', fontsize=32)

            boundary_nodes = edge_index_bound[1, boundary_neighbors_idx]
            for ax_idx, j in enumerate(boundary_nodes):
                ax_idx = ax_idx + 1
                bound_idx = j - num_nodes_int
                x_j_gt = X_bound[:, bound_idx]

                axes[ax_idx].plot(timestamps, x_j_gt, label='Dirichlet Boundary', c='black',
                                  linestyle='dashed', linewidth=8)
                axes[ax_idx].set_title(f'Boundary Node {j}', fontsize=48)
                axes[ax_idx].set_facecolor("gainsboro")
                axes[ax_idx].tick_params(axis='both', which='major', labelsize=24)
                axes[ax_idx].legend(loc='upper right', fontsize=32)
                axes[ax_idx].set_xlabel("Time Step k", fontsize=32)
                axes[ax_idx].set_ylabel("Boundary", fontsize=32)
                axes[ax_idx].legend()
                axes[ax_idx].legend(loc='upper left', fontsize=32)
                axes[ax_idx].grid(True, color='gray', linewidth=0.5)  # Add grid lines
            plt.tight_layout()
            if visualization_dir:
                plt.savefig(os.path.join(visualization_dir, f"state_{i}_withBound_{j}.png"), dpi=150)
            else:
                plt.show()
            plt.clf()
            plt.close()


def visualize_state_ocp(X_int_bigoc,
                        timestamps,
                        x_int_des,
                        num_nodes_int,
                        visualization_dir=None):

    timestamps = [i for i in range(len(timestamps))]
    fig, axes = plt.subplots(num_nodes_int, 1, figsize=(24, 10 * num_nodes_int))
    for i in range(num_nodes_int):
        x_int_bignode_i = X_int_bigoc[:, i]

        axes[i].set_title(f"Interior node {i}", fontsize=48)
        axes[i].plot(timestamps, [x_int_des[i] for _ in range(len(timestamps))], label='desired state',
                     c='black', linewidth=6, linestyle='--')
        axes[i].plot(timestamps, x_int_bignode_i, label='BigNode + u_BigOC', c='blue', linewidth=4)
        axes[i].set_facecolor("whitesmoke")
        axes[i].grid(True, color='gray', linewidth=0.5)  # Add grid lines
        axes[i].set_xlabel("Time Step k", fontsize=32)
        axes[i].set_ylabel("State", fontsize=32)
        axes[i].tick_params(axis='both', which='major', labelsize=24)
        axes[i].legend(loc='lower right', fontsize=32)
    plt.tight_layout()
    if visualization_dir is not None:
        plt.savefig(os.path.join(visualization_dir, "state.png"), dpi=150)
    else:
        plt.show()
    plt.clf()
    plt.close()


def visualize_ctrl(U_bigoc,
                   timestamps,
                   num_nodes_int,
                   num_nodes_bound,
                   num_nodes_ctrl,
                   visualization_dir):
    timestamps = [i for i in range(len(timestamps))]
    fig, axes = plt.subplots(num_nodes_ctrl, 1, dpi=150, figsize=(24, 10))
    if num_nodes_ctrl == 1:
        axes = [axes]
    for i in range(num_nodes_ctrl):
        u_bigoc_i = U_bigoc[:, i]
        axes[i].set_title(f"Control node {i + num_nodes_int + num_nodes_bound}", fontsize=48)
        axes[i].step(timestamps[:-1], u_bigoc_i, label='u_BigOC', c='red', where='post', linewidth=4)
        axes[i].set_facecolor("whitesmoke")
        axes[i].grid(True, color='gray', linewidth=0.5)
        axes[i].set_xlabel("Time Step k", fontsize=32)
        axes[i].set_ylabel("Control", fontsize=32)
        axes[i].tick_params(axis='both', which='major', labelsize=24)
        axes[i].legend()
        axes[i].legend(loc='upper right', fontsize=32)
    plt.tight_layout()
    if visualization_dir is not None:
        plt.savefig(os.path.join(visualization_dir, "control.png"), dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()
