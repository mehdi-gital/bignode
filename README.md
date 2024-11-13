# Inverse Boundary Value and Optimal Control Problems on Graphs: A Neural and Numerical Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2206.02911-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2206.02911)

A general setup for deterministic system identification problems on graphs with Dirichlet and Neumann boundary conditions is introduced. When control nodes are available along the boundary, we apply a discretize-then-optimize method to estimate an optimal control. A key piece in the present architecture is our boundary injected message passing neural network. This will produce more accurate predictions that are considerably more stable in proximity of the boundary. Also, a regularization technique based on graphical distance is introduced that helps with stabilizing the predictions at nodes far from the boundary.

# Setup

System requirements:
- Python <= 3.12
- Pip package manager

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

For system identification, dataset selection, hyperparameter optimization, go to `sysid/graph_sysid.py`. Once set, run it using:

```
python sysid/graph_sysid.py
```

$ Demo

A notebook tutorial on system identification in the linear case

```
sysid/system_identification_linear_demo.ipynb
```



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
