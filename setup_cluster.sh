#!/bin/bash

# setup_cluster.sh
# Run this on your Linux/CUDA cluster node to set up the full performance environment.

echo "Starting Cluster Environment Setup..."

# 1. Create a fresh venv (recommended)
python3 -m venv .venv_cluster
source .venv_cluster/bin/activate

# 2. Install PyTorch with CUDA 12.4 support (Required for Flash Attention)
echo "Installing PyTorch (CUDA 12.4)..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install Flash STU (Standard install works on Linux)
# This will automatically pull in triton, which works on Linux.
echo "Installing Flash STU..."
pip install git+https://github.com/hazan-lab/flash-stu-2.git

# 4. Install Optional Performance Dependencies
echo "Installing Flash Attention (this may take a while)..."
export MAX_JOBS=4
pip install flash-attn --no-build-isolation

echo "Installing Flash FFT Conv..."
pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
pip install git+https://github.com/HazyResearch/flash-fft-conv.git

# 5. Install other project dependencies
# Use the same requirements filter, but you might want to review it for linux specifics if needed.
# For now, we assume requirements_filtered.txt contains platform-agnostic deps.
if [ -f requirements_filtered.txt ]; then
    echo "Installing remaining dependencies..."
    pip install -r requirements_filtered.txt
fi

echo "Cluster setup complete!"
