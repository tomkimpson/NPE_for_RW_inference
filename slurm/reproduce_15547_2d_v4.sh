#!/bin/bash

#SBATCH --job-name=reproduce_15547_2d_v4
#SBATCH --output=slurm/outputs/reproduce_15547_2d_v4_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

# 2D CNN Reproduction script for workflow_20250827_155447 - Optimized v4
# This script uses significantly heavier settings for maximum accuracy
# Based on v3 but with enhanced computational budget and parameters
# Original timestamp: 2025-08-27 15:54:47.792687

echo "Starting heavily optimized 2D CNN reproduction of workflow_20250827_155447 (v4)..."
echo "Enhanced configuration for maximum accuracy:"
echo "  Lx: 200, Ly: 50, T: 100"
echo "  theta_true: [0.3, 0.7]"
echo "  SNPE rounds: 12, samples per round: 8000 (96k total)"
echo "  Enhanced network: 512 hidden features, 8 transforms"
echo "  Lower learning rate: 3e-5 for optimal convergence"
echo "  Larger batch size: 1024 for better gradient estimates"
echo "  2D spatial data with CNN processing"

time python src/main.py \
    --use_2d_data \
    --Lx 200 \
    --Ly 50 \
    --T 100 \
    --initial_region_half_width 25 \
    --max_epochs 200 \
    --batch_size 1024 \
    --learning_rate 3e-5 \
    --hidden_features 512 \
    --num_transforms 8 \
    --validation_fraction 0.1 \
    --stop_after_epochs 40 \
    --theta_true 0.3 0.7 \
    --use_snpe \
    --snpe_rounds 12 \
    --samples_per_round 8000 \
    --convergence_threshold 0.01 \
    --seed 42

echo "Heavily optimized 2D CNN reproduction workflow (v4) completed!"