#!/bin/bash
#
# ABC baseline: SMC-ABC inference for all models.
# Usage:  sbatch slurm/run_abc.sh <MODEL>
# where MODEL is one of: original, A, B, C
#

#SBATCH --job-name=npe_abc
#SBATCH --output=slurm/outputs/npe_abc_%j.txt
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Model name from positional argument
MODEL="${1:-A}"

# Activate conda env
source ~/.bashrc
conda activate NPE_LV

# Pin each process to 1 BLAS/OpenMP thread
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "ABC RUN: model=${MODEL}  n_workers=8  n_simulations=50000"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================"

python src/main.py \
    --model "${MODEL}" \
    --method abc \
    --abc_num_particles 500 \
    --abc_num_simulations 50000 \
    --abc_num_initial_pop 2000 \
    --abc_epsilon_decay 0.5 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --n_pred_samples 500 \
    --n_workers 8

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
