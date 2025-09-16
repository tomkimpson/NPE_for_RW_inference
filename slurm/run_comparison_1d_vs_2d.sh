#!/bin/bash

#SBATCH --job-name=snpe_1d_vs_2d_comparison
#SBATCH --output=slurm/outputs/snpe_comparison_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV

# This script runs both 1D and 2D versions with identical parameters
# for direct comparison of inference quality

echo "=========================================="
echo "Starting 1D vs 2D NPE Comparison Study"
echo "=========================================="

# Common parameters
COMMON_PARAMS="--use_snpe --snpe_rounds 6 --samples_per_round 3000 \
               --max_epochs 150 --stop_after_epochs 12 \
               --convergence_threshold 0.002 \
               --learning_rate 1e-4 \
               --batch_size 128 \
               --Lx 80 --Ly 40 --T 100 --initial_region_half_width 10 \
               --num_samples 6000 \
               --n_pred_samples 2000 \
               --theta_true 0.4 0.6"

echo "Configuration for comparison:"
echo "Lattice: 80x40, Time steps: 100"
echo "True parameters: U=0.4, P=0.6"
echo "SNPE rounds: 6, Samples per round: 3000"
echo ""

# Run 1D version
echo "==========================================="
echo "PHASE 1: Running 1D NPE (Column Counts)"
echo "==========================================="
echo "Start time: $(date)"

time python src/main.py $COMMON_PARAMS \
      --hidden_features 128 --num_transforms 6 \
      --output_dir results/comparison_1d_$(date +%Y%m%d_%H%M%S)

echo "1D NPE completed at: $(date)"
echo ""

# Run 2D version
echo "==========================================="
echo "PHASE 2: Running 2D NPE (Spatial CNN)"
echo "==========================================="
echo "Start time: $(date)"

time python src/main.py $COMMON_PARAMS --use_2d_data \
      --hidden_features 128 --num_transforms 6 \
      --batch_size 64 \
      --output_dir results/comparison_2d_$(date +%Y%m%d_%H%M%S)

echo "2D NPE completed at: $(date)"
echo ""

echo "=========================================="
echo "COMPARISON STUDY COMPLETED"
echo "=========================================="
echo "Results saved in:"
echo "- results/comparison_1d_* (1D column-based NPE)"
echo "- results/comparison_2d_* (2D spatial CNN NPE)"
echo ""
echo "Compare the following metrics:"
echo "1. Posterior accuracy (credible interval coverage)"
echo "2. Posterior precision (CI width)"
echo "3. Training efficiency (convergence speed)"
echo "4. Prediction quality (predictive intervals)"