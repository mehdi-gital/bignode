# Inverse Boundary Value and Optimal Control Problems on Graphs: A Neural and Numerical Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2206.02911-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2206.02911)

A general setup for deterministic system identification problems on graphs with Dirichlet and Neumann boundary conditions is introduced. When control nodes are available along the boundary, we apply a discretize-then-optimize method to estimate an optimal control. A key piece in the present architecture is our boundary injected message passing neural network. This will produce more accurate predictions that are considerably more stable in proximity of the boundary. Also, a regularization technique based on graphical distance is introduced that helps with stabilizing the predictions at nodes far from the boundary.

<div style="text-align: center;">
<img src="notebooks\imgs\discrete_boundary.png" width="1000" />
</div>

The main contribution of this project is a _Boundary Injected Message Passing Layer_ for a structural approach to enforce boundary values visualied as follows:

<div style="text-align: center;">
<img src="notebooks\imgs\boundary_mp_sub.png" width="1000" />
</div>

See `BoundaryInjectedMessagePassingLayer` in both `core/bignode_no_ctrl.py` and `core/bignode_with_ctrl.py` for details. 

# Setup

System requirements:
- Python <= 3.12
- Pip package manager or alternative

Simply run:
```
./setup.sh
```

Installation steps:

```
git clone https://github.com/mehdi-gital/bignode.git
cd bignode
python -m venv .venv
source .venv/bin/activate
```

# Usage

For system identification, dataset selection, hyperparameter optimization, go to `scripts/graph_sysid.py`. Once set, run it using:

```
python scripts/graph_sysid.py
```

For optimal control go to `scripts/graph_ocp.py`, and run:

```
python scripts/graph_ocp.py
```

<div style="text-align: center;">
<img src="notebooks\imgs\lin_ctrl.png" width="1000" />
</div>

# Demo

Below are some step by step walkthroughs with a mix of concept and code:

- [System Identification Tutorial (linear)](notebooks/system_identification_linear_demo.ipynb)
<!-- - [System Identification Tutorial (nonlinear)](notebooks/system_identification_nonlinear_demo.ipynb)
- [Optimal Control (linear)](notebooks/optimal_control_linear_demo.ipynb) -->

<div style="text-align: center;">
<img src="notebooks\imgs\nonlin.png" width="1000" />
</div>

# Paper


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

# Contact

Feel free to reach out with any questions/comment. 

- `mgarrous@alumni.uwo.ca` 
- `amirhossein.nouranizadeh@gmail.com`