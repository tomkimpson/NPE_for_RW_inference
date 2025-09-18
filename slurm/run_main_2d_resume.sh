#!/bin/bash

#SBATCH --job-name=snpe_random_walk_2d_resume
#SBATCH --output=slurm/outputs/snpe_random_walk_2d_resume_%j.txt
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

# Resume from existing workflow directory
RESUME_DIR="results/workflow_20250917_144333"

echo "Starting 2D NPE Random Walk RESUME workflow..."
echo "Resuming from: $RESUME_DIR"
echo "Configuration: 2D spatial data with CNN processing"
echo "Lattice size: 200x50, Time steps: 100"
echo "Target SNPE rounds: 10 (resuming from completed rounds)"

time python src/main.py --use_2d_data --use_snpe \
      --resume_from_dir "$RESUME_DIR" \
      --snpe_rounds 10 --samples_per_round 10000 \
      --max_epochs 300 --stop_after_epochs 10 \
      --hidden_features 128 --num_transforms 8 \
      --convergence_threshold 0.001 \
      --learning_rate 1e-4 \
      --batch_size 128 \
      --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
      --num_samples 10000 \
      --n_pred_samples 2000 \
      --theta_true 0.3 0.7 \
      --seed 42

echo "2D NPE resume workflow completed!"