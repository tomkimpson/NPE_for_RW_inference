#!/bin/bash
#
# Enhanced 2D NPE production run with CNN embedding.
#
# Fixes hidden_features alignment: CNN output_dim and NSF hidden_features
# both set to 256 (previous runs had CNN=256 but NSF=128 due to CLI override).
#
# Usage:
#   sbatch slurm/run_production_2d_enhanced.sh <MODEL> [N_SAMPLES] [--seed SEED]
#
# Examples:
#   sbatch slurm/run_production_2d_enhanced.sh A          # 50k samples, seed=42
#   sbatch slurm/run_production_2d_enhanced.sh A 100000   # 100k samples
#   sbatch slurm/run_production_2d_enhanced.sh A 50000 --seed 123  # different seed
#

#SBATCH --job-name=npe_2d_enh
#SBATCH --output=slurm/outputs/npe_2d_enh_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Parse arguments
MODEL="${1:-A}"
N_SAMPLES="${2:-50000}"
shift 2 2>/dev/null || shift $# 2>/dev/null
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
echo "ENHANCED 2D NPE RUN"
echo "  model=${MODEL}  n_samples=${N_SAMPLES}  n_workers=16"
echo "  hidden_features=256 (aligned with CNN output_dim)"
echo "  Extra args: ${EXTRA_ARGS}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================"

python src/main.py \
    --model "${MODEL}" \
    --use_2d_data \
    --n_samples "${N_SAMPLES}" \
    --max_epochs 100 --stop_after_epochs 20 \
    --hidden_features 256 --num_transforms 8 \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 5000 \
    --n_pred_samples 500 \
    --n_workers 16 \
    ${EXTRA_ARGS}

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
