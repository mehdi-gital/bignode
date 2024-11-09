#!/bin/bash

# Install all dependencies listed in requirements.txt, using the PyG-specific wheel link for PyTorch 2.3.0 compatibility
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
