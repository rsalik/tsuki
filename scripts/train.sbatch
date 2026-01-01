#!/bin/bash
#SBATCH --job-name=tsuki-pong
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=rs3216@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup

# ============================================
# World Model Training - SLURM Job Script
# ============================================

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# Configuration file (can be overridden with --export=CONFIG=path)
CONFIG=${CONFIG:-"configs/pong.yaml"}
echo "Using config: $CONFIG"

# Activate virtual environment
source .venv/bin/activate  # UPDATE if using different venv

# Ensure dependencies are installed
pip install -q d4rl-atari gym tqdm pyyaml

# Create necessary directories
mkdir -p logs checkpoints

# Run training
echo ""
echo "Starting training..."
echo "============================================"

python src/main.py --config "$CONFIG"

echo ""
echo "============================================"
echo "Training complete!"
echo "End time: $(date)"
echo "============================================"
