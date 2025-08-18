#!/bin/bash
# Setup script for SACRED environment with conda

echo "==================================="
echo "SACRED Environment Setup (Conda)"
echo "==================================="

# Environment name
ENV_NAME="sacred"

# Create conda environment with Python 3.10
echo "Creating conda environment..."
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch (CPU version)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install RDKit (much easier with conda)
echo "Installing RDKit..."
conda install -c conda-forge rdkit -y

# Install scientific computing packages
echo "Installing scientific packages..."
conda install numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y

# Install additional packages with pip
echo "Installing additional packages..."
pip install transformers>=4.30.0
pip install tqdm pyyaml selfies

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Detected PyTorch version: $TORCH_VERSION"

# Install PyG dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION%+*}+cpu.html
pip install torch-geometric

echo "==================================="
echo "Setup complete!"
echo "Activate environment with: conda activate $ENV_NAME"
echo "To deactivate: conda deactivate"
echo "To remove environment: conda env remove -n $ENV_NAME"
echo "==================================="