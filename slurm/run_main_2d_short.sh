#!/bin/bash

#SBATCH --job-name=snpe_random_walk_2d
#SBATCH --output=slurm/outputs/snpe_random_walk_2d_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

#Run command with 2D spatial data processing
# Note: Using CNN for 2D spatial data requires more memory and compute
# Adjusted parameters for 2D spatial learning:
# - Increased memory allocation (32G vs 16G)
# - Increased CPUs for CNN processing
# - Adjusted batch size for 2D data
# - Using smaller spatial dimensions initially (100x50 vs 200x50)

echo "Starting 2D NPE Random Walk workflow..."
echo "Configuration: 2D spatial data with CNN processing"
echo "Lattice size: 100x50, Time steps: 100"
echo "SNPE rounds: 8, Samples per round: 5000"

time python src/main.py --use_2d_data --use_snpe \
      --snpe_rounds 5 --samples_per_round 10000 \
      --max_epochs 300 --stop_after_epochs 10 \
      --hidden_features 128 --num_transforms 8 \
      --convergence_threshold 0.001 \
      --learning_rate 1e-4 \
      --batch_size 128 \
      --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
      --num_samples 10000 \
      --n_pred_samples 2000

echo "2D NPE workflow completed!"