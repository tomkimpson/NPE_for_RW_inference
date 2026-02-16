#!/bin/bash
#
# Production run: full NPE workflow with 10k training samples.
# Usage:  sbatch slurm/run_production.sh <MODEL> [--use_2d_data]
# where MODEL is one of: original, A, B, C
#
# Examples:
#   sbatch slurm/run_production.sh A               # 1D NPE
#   sbatch slurm/run_production.sh A --use_2d_data  # 2D NPE with CNN
#

#SBATCH --job-name=npe_prod
#SBATCH --output=slurm/outputs/npe_prod_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Model name from positional argument
MODEL="${1:-A}"
shift || true
EXTRA_ARGS="$@"

# Activate conda env
source ~/.bashrc
conda activate NPE_LV

# Pin each process to 1 BLAS/OpenMP thread so multiprocessing workers
# use separate cores instead of all contending on the same thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "PRODUCTION RUN: model=${MODEL}  n_workers=8  n_samples=10000"
echo "Extra args: ${EXTRA_ARGS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================"

python src/main.py \
    --model "${MODEL}" \
    --n_samples 10000 \
    --max_epochs 100 --stop_after_epochs 20 \
    --hidden_features 128 --num_transforms 8 \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 5000 \
    --n_pred_samples 500 \
    --n_workers 8 \
    ${EXTRA_ARGS}

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
