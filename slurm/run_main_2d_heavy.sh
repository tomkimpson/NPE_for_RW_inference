#!/bin/bash

#SBATCH --job-name=snpe_2d_heavy
#SBATCH --output=slurm/outputs/snpe_2d_heavy_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

#Run command with 2D spatial data processing - HEAVY VERSION for production
# Large lattice, many samples for high-quality results with full 2D spatial resolution
# This configuration takes advantage of the CNN's ability to process large spatial data

echo "Starting 2D NPE Random Walk workflow (HEAVY VERSION)..."
echo "Configuration: 2D spatial data with CNN processing"
echo "Lattice size: 200x100, Time steps: 200"
echo "SNPE rounds: 12, Samples per round: 8000"
echo "WARNING: This is a computationally intensive run!"

time python src/main.py --use_2d_data --use_snpe \
      --snpe_rounds 12 --samples_per_round 8000 \
      --max_epochs 400 --stop_after_epochs 20 \
      --hidden_features 256 --num_transforms 10 \
      --convergence_threshold 0.0005 \
      --learning_rate 2e-5 \
      --batch_size 32 \
      --Lx 200 --Ly 100 --T 200 --initial_region_half_width 25 \
      --num_samples 15000 \
      --n_pred_samples 5000

echo "2D NPE heavy workflow completed!"