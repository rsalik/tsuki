#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting environment setup...${NC}"

# Check if python3.12 is available
if [ ! -f /Users/rsalik/.pyenv/versions/3.12.8/bin/python ]; then
    echo -e "${RED}Error: python3.12 could not be found at /Users/rsalik/.pyenv/versions/3.12.8/bin/python.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment .venv_3.12...${NC}"
/Users/rsalik/.pyenv/versions/3.12.8/bin/python -m venv .venv_3.12

# Activate virtual environment
source .venv_3.12/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install specific versions
echo -e "${GREEN}Installing PyTorch 2.4.1...${NC}"
# MacOS usually needs specific pip commands for torch, but standard pip install torch==2.4.1 should grab the mac wheels if available.
pip install torch==2.4.1

echo -e "${GREEN}Attempting to install Triton 3.0.0...${NC}"
if pip install triton==3.0.0; then
    echo -e "${GREEN}Triton installed successfully.${NC}"
else
    echo -e "${YELLOW}Warning: Triton installation failed. This is expected on MacOS as Triton is primarily for CUDA.${NC}"
    echo -e "${YELLOW}Continuing without Triton...${NC}"
fi

# Install other dependencies
echo -e "${GREEN}Installing remaining dependencies from requirements_filtered.txt...${NC}"
if [ -f requirements_filtered.txt ]; then
    pip install -r requirements_filtered.txt
else
    echo -e "${RED}Error: requirements_filtered.txt not found!${NC}"
fi

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}To activate, run: source .venv_3.12/bin/activate${NC}"
