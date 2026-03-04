#!/bin/bash
#
# Light test run to verify the pipeline works end-to-end.
# Usage:  sbatch slurm/run_model_light.sh <MODEL>
# where MODEL is one of: original, A, B, C
#

#SBATCH --job-name=npe_light
#SBATCH --output=slurm/outputs/npe_light_%A_%a.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Model name from first positional argument (default: original)
MODEL="${1:-original}"

# Activate conda env
source ~/.bashrc
conda activate NPE_LV

echo "Running LIGHT test for model: ${MODEL}"

time python src/main.py \
    --model "${MODEL}" \
    --n_samples 500 \
    --max_epochs 20 --stop_after_epochs 10 \
    --hidden_features 128 --num_transforms 8 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 200 \
    --n_pred_samples 50 \
    --n_workers 8
