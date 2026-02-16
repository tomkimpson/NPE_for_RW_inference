#!/bin/bash
#
# Rerun 2D NPE with bias fixes using existing training data.
#

#SBATCH --job-name=npe_2d_fix2
#SBATCH --output=slurm/outputs/npe_2d_fix2_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

mkdir -p slurm/outputs

source ~/.bashrc
conda activate NPE_LV

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "2D NPE WITH BIAS FIXES (using existing data)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================"

python src/main.py \
    --model A \
    --use_2d_data \
    --cnn_auxiliary_features \
    --n_samples 50000 \
    --skip_data \
    --data_path results/workflow_A_npe2d_20260206_160159/training_data.pkl \
    --max_epochs 100 --stop_after_epochs 20 \
    --hidden_features 256 --num_transforms 8 \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 5000 \
    --n_pred_samples 500 \
    --n_workers 16

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
