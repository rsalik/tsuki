#!/bin/bash

# ============================================
# Unified Environment Setup Script
# Handles both local development and cluster deployment
# ============================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect environment
detect_environment() {
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "macos"
    elif command -v nvidia-smi &> /dev/null; then
        echo "cluster"
    else
        echo "linux"
    fi
}

ENV_TYPE=$(detect_environment)

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Tsuki World Model - Environment Setup   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Detected environment: ${GREEN}${ENV_TYPE}${NC}"
echo ""

# ============================================
# 0. Load Modules (Cluster Only)
# ============================================
if [[ "$ENV_TYPE" == "cluster" ]]; then
    echo -e "${GREEN}Loading cluster modules...${NC}"
    
    # Princeton Della/Tiger cluster modules
    if command -v module &> /dev/null; then
        # Try to load Python 3.12 first, then 3.11
        if module avail python/3.12 2>&1 | grep -q "python/3.12"; then
            module load python/3.12
            echo -e "${GREEN}Loaded Python 3.12 module${NC}"
        elif module avail anaconda3/2024 2>&1 | grep -q "anaconda3"; then
            module load anaconda3/2024.10
            echo -e "${GREEN}Loaded Anaconda 2024.10 module${NC}"
        elif module avail python/3.11 2>&1 | grep -q "python/3.11"; then
            module load python/3.11
            echo -e "${GREEN}Loaded Python 3.11 module${NC}"
        else
            echo -e "${YELLOW}Warning: Could not find Python 3.11+ module.${NC}"
            echo -e "${YELLOW}Available Python modules:${NC}"
            module avail python 2>&1 | head -20
            echo ""
            echo -e "${RED}Please load a Python 3.11+ module manually and re-run.${NC}"
            echo -e "${RED}Example: module load anaconda3/2024.10${NC}"
            exit 1
        fi
        
        # Also load CUDA module
        if module avail cudatoolkit/12 2>&1 | grep -q "cudatoolkit"; then
            module load cudatoolkit/12.4
            echo -e "${GREEN}Loaded CUDA 12.4 module${NC}"
        fi
    fi
fi

# ============================================
# 1. Create Virtual Environment
# ============================================
if [[ "$ENV_TYPE" == "macos" ]]; then
    VENV_NAME=".venv"
    
    # Check for Python 3.12
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif [ -f /Users/rsalik/.pyenv/versions/3.12.8/bin/python ]; then
        PYTHON_CMD="/Users/rsalik/.pyenv/versions/3.12.8/bin/python"
    else
        echo -e "${YELLOW}Warning: Python 3.12 not found, using default python3${NC}"
        PYTHON_CMD="python3"
    fi
else
    VENV_NAME=".venv"
    PYTHON_CMD="python3"
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo -e "${GREEN}Using Python: $PYTHON_CMD (version $PYTHON_VERSION)${NC}"

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    echo -e "${RED}Error: Python 3.11+ is required. Found: $PYTHON_VERSION${NC}"
    echo -e "${RED}Please ensure Python 3.11+ is available.${NC}"
    exit 1
fi

echo -e "${GREEN}Creating virtual environment: ${VENV_NAME}${NC}"
$PYTHON_CMD -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# ============================================
# 2. Install PyTorch
# ============================================
echo ""
echo -e "${GREEN}Installing PyTorch...${NC}"

if [[ "$ENV_TYPE" == "cluster" ]]; then
    # CUDA 12.4 for cluster with GPU
    echo -e "${BLUE}Installing PyTorch with CUDA 12.4 support...${NC}"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    # CPU/MPS version for local development
    echo -e "${BLUE}Installing PyTorch (CPU/MPS)...${NC}"
    pip install torch
fi

# ============================================
# 3. Install Flash STU and Dependencies
# ============================================
echo ""
echo -e "${GREEN}Installing Flash STU...${NC}"

if [[ "$ENV_TYPE" == "cluster" ]]; then
    # Full installation on cluster
    echo -e "${BLUE}Installing Flash STU with full performance dependencies...${NC}"
    
    # Install Flash STU from GitHub
    pip install git+https://github.com/hazan-lab/flash-stu-2.git
    
    # Install Flash Attention (optional but recommended)
    echo -e "${GREEN}Installing Flash Attention (this may take a while)...${NC}"
    export MAX_JOBS=4
    if pip install flash-attn --no-build-isolation; then
        echo -e "${GREEN}Flash Attention installed successfully.${NC}"
    else
        echo -e "${YELLOW}Warning: Flash Attention installation failed. Continuing without it.${NC}"
    fi
    
    # Install Flash FFT Conv (optional)
    echo -e "${GREEN}Installing Flash FFT Conv...${NC}"
    if pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv && \
       pip install git+https://github.com/HazyResearch/flash-fft-conv.git; then
        echo -e "${GREEN}Flash FFT Conv installed successfully.${NC}"
    else
        echo -e "${YELLOW}Warning: Flash FFT Conv installation failed. Continuing without it.${NC}"
    fi
else
    # Lightweight installation for local dev (from local dependencies folder)
    echo -e "${BLUE}Installing Flash STU (lightweight mode for local dev)...${NC}"
    
    if [ -d "dependencies/flash-stu-2" ]; then
        pip install -e dependencies/flash-stu-2
        echo -e "${GREEN}Flash STU installed from local dependencies.${NC}"
    else
        echo -e "${YELLOW}Warning: dependencies/flash-stu-2 not found.${NC}"
        echo -e "${YELLOW}Flash STU will not be available locally (requires CUDA anyway).${NC}"
    fi
    
    # Skip Triton on macOS (not supported)
    if [[ "$ENV_TYPE" == "macos" ]]; then
        echo -e "${YELLOW}Skipping Triton (not supported on macOS).${NC}"
    fi
fi

# ============================================
# 4. Install D4RL-Atari and Gym
# ============================================
echo ""
echo -e "${GREEN}Installing D4RL-Atari and Gym...${NC}"

pip install gym gymnasium
if pip install d4rl-atari; then
    echo -e "${GREEN}D4RL-Atari installed successfully.${NC}"
else
    echo -e "${YELLOW}Warning: D4RL-Atari installation failed.${NC}"
    echo -e "${YELLOW}You may need to install it manually: pip install d4rl-atari${NC}"
fi

# ============================================
# 5. Install Project Dependencies
# ============================================
echo ""
echo -e "${GREEN}Installing project dependencies...${NC}"

# Core dependencies that work on all platforms
pip install pyyaml tqdm tensorboard

# Install from requirements file if exists
if [ -f "requirements_filtered.txt" ]; then
    echo -e "${GREEN}Installing from requirements_filtered.txt...${NC}"
    pip install -r requirements_filtered.txt || echo -e "${YELLOW}Some requirements failed to install.${NC}"
fi

# ============================================
# 6. Create Required Directories
# ============================================
echo ""
echo -e "${GREEN}Creating project directories...${NC}"
mkdir -p checkpoints logs configs

# ============================================
# 7. Verify Installation
# ============================================
echo ""
echo -e "${GREEN}Verifying installation...${NC}"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    from flash_stu import FlashSTU, FlashSTUBlock
    print('Flash STU: ✓ Installed')
except ImportError as e:
    print(f'Flash STU: ✗ Not available ({e})')

try:
    import d4rl_atari
    print('D4RL-Atari: ✓ Installed')
except ImportError:
    print('D4RL-Atari: ✗ Not available')

try:
    import yaml
    print('PyYAML: ✓ Installed')
except ImportError:
    print('PyYAML: ✗ Not available')
"

# ============================================
# Done!
# ============================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "  ${YELLOW}source ${VENV_NAME}/bin/activate${NC}"
echo ""

if [[ "$ENV_TYPE" == "cluster" ]]; then
    echo -e "To train on cluster:"
    echo -e "  ${YELLOW}sbatch scripts/train.sbatch${NC}"
else
    echo -e "To test locally (without CUDA):"
    echo -e "  ${YELLOW}python src/main.py --config configs/pong.yaml --device cpu --epochs 1${NC}"
fi
