# BigNode

This repo contains our supporting code for the following manuscript:

A general setup for deterministic system identification problems on graphs with Dirichlet and Neumann boundary conditions is introduced. When control nodes are available along the boundary, we apply a discretize-then-optimize method to estimate an optimal control. A key piece in the present architecture is our boundary injected message passing neural network. This will produce more accurate predictions that are considerably more stable in proximity of the boundary. Also, a regularization technique based on graphical distance is introduced that helps with stabilizing the predictions at nodes far from the boundary.

```
@misc{garrousian2024inverseboundaryvalueoptimal,
      title={Inverse Boundary Value and Optimal Control Problems on Graphs: A Neural and Numerical Synthesis}, 
      author={Mehdi Garrousian and Amirhossein Nouranizadeh},
      year={2024},
      eprint={2206.02911},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2206.02911}, 
}
```
