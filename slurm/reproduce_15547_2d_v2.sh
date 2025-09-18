#!/bin/bash

#SBATCH --job-name=reproduce_15547_2d
#SBATCH --output=slurm/outputs/reproduce_15547_2d_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

# 2D CNN Reproduction script for workflow_20250827_155447
# This script reproduces the exact configuration from results/workflow_20250827_155447/config.txt
# but adapted for 2D spatial data processing with CNN functionality
# Original timestamp: 2025-08-27 15:54:47.792687

echo "Starting 2D CNN reproduction of workflow_20250827_155447..."
echo "Configuration parameters adapted from original config.txt:"
echo "  Lx: 200, Ly: 50, T: 100"
echo "  theta_true: [0.3, 0.7]"
echo "  SNPE rounds: 2, samples: 10000"
echo "  2D spatial data with CNN processing"

time python src/main.py \
    --use_2d_data \
    --Lx 200 \
    --Ly 50 \
    --T 100 \
    --initial_region_half_width 25 \
    --n_samples 10000 \
    --max_epochs 100 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --hidden_features 128 \
    --num_transforms 5 \
    --validation_fraction 0.1 \
    --stop_after_epochs 30 \
    --theta_true 0.3 0.7 \
    --num_samples 6000 \
    --use_snpe \
    --snpe_rounds 6 \
    --samples_per_round 5000 \
    --convergence_threshold 0.01 \
    --seed 42

echo "2D CNN reproduction workflow completed!"