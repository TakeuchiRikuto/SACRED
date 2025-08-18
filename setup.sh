#!/bin/bash
# Setup script for SACRED environment

echo "==================================="
echo "SACRED Environment Setup"
echo "==================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic requirements
echo "Installing basic requirements..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.30.0
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn tqdm pyyaml
pip install selfies

# Install RDKit (try different methods)
echo "Installing RDKit..."
pip install rdkit-pypi || pip install rdkit || echo "Please install RDKit via conda"

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Detected PyTorch version: $TORCH_VERSION"

# Install PyG dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION%+*}+cpu.html
pip install torch-geometric

echo "==================================="
echo "Setup complete!"
echo "Activate environment with: source venv/bin/activate"
echo "==================================="