#!/bin/bash

#SBATCH --job-name=reproduce_15547
#SBATCH --output=slurm/outputs/reproduce_15547_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

# Reproduction script for workflow_20250827_155447
# This script reproduces the exact configuration from results/workflow_20250827_155447/config.txt
# Original timestamp: 2025-08-27 15:54:47.792687

echo "Starting reproduction of workflow_20250827_155447..."
echo "Configuration parameters from original config.txt:"
echo "  Lx: 200, Ly: 50, T: 100"
echo "  theta_true: [0.3, 0.7]"
echo "  SNPE rounds: 2, samples: 10000"
echo "  Device: cuda (Tesla P100-PCIE-12GB)"

time python src/main.py \
    --Lx 200 \
    --Ly 50 \
    --T 100 \
    --initial_region_half_width 25 \
    --n_samples 10000 \
    --max_epochs 100 \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --hidden_features 128 \
    --num_transforms 5 \
    --validation_fraction 0.1 \
    --stop_after_epochs 20 \
    --theta_true 0.3 0.7 \
    --num_samples 5000 \
    --use_snpe \
    --snpe_rounds 2 \
    --convergence_threshold 0.01 \
    --seed 42

echo "Reproduction workflow completed!"