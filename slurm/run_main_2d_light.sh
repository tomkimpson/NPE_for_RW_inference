#!/bin/bash

#SBATCH --job-name=snpe_2d_light
#SBATCH --output=slurm/outputs/snpe_2d_light_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

#Run command with 2D spatial data processing - LIGHT VERSION for testing
# Smaller lattice, fewer samples for quick validation of 2D functionality

echo "Starting 2D NPE Random Walk workflow (LIGHT VERSION)..."
echo "Configuration: 2D spatial data with CNN processing"
echo "Lattice size: 50x25, Time steps: 50"
echo "SNPE rounds: 5, Samples per round: 2000"

time python src/main.py --use_2d_data --use_snpe \
      --snpe_rounds 5 --samples_per_round 2000 \
      --max_epochs 100 --stop_after_epochs 10 \
      --hidden_features 64 --num_transforms 4 \
      --convergence_threshold 0.005 \
      --learning_rate 1e-4 \
      --batch_size 128 \
      --Lx 50 --Ly 25 --T 50 --initial_region_half_width 6 \
      --num_samples 5000 \
      --n_pred_samples 1000

echo "2D NPE light workflow completed!"