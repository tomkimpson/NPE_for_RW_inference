#!/bin/bash

#SBATCH --job-name=reproduce_15547_2d_v3
#SBATCH --output=slurm/outputs/reproduce_15547_2d_v3_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

# 2D CNN Reproduction script for workflow_20250827_155447 - Optimized v3
# This script uses heavier settings to reduce bias in 'P' parameter inference
# Based on results/workflow_20250827_155447/config.txt but with enhanced parameters
# Original timestamp: 2025-08-27 15:54:47.792687

echo "Starting optimized 2D CNN reproduction of workflow_20250827_155447 (v3)..."
echo "Enhanced configuration for bias reduction:"
echo "  Lx: 200, Ly: 50, T: 100"
echo "  theta_true: [0.3, 0.7]"
echo "  SNPE rounds: 8, samples per round: 6000"
echo "  Enhanced network: 256 hidden features"
echo "  Lower learning rate: 5e-5 for better convergence"
echo "  2D spatial data with CNN processing"

time python src/main.py \
    --use_2d_data \
    --Lx 200 \
    --Ly 50 \
    --T 100 \
    --initial_region_half_width 25 \
    --max_epochs 150 \
    --batch_size 512 \
    --learning_rate 5e-5 \
    --hidden_features 256 \
    --num_transforms 5 \
    --validation_fraction 0.1 \
    --stop_after_epochs 40 \
    --theta_true 0.3 0.7 \
    --use_snpe \
    --snpe_rounds 8 \
    --samples_per_round 6000 \
    --convergence_threshold 0.01 \
    --seed 42

echo "Optimized 2D CNN reproduction workflow (v3) completed!"