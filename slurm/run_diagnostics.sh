#!/bin/bash
#
# Run SBI diagnostics (SBC + TARP) for a trained NPE model.
# Usage:  sbatch slurm/run_diagnostics.sh <MODEL> <MODEL_PATH>
#
# Examples:
#   sbatch slurm/run_diagnostics.sh original results/workflow_original_npe_20260204_225605/npe_model.pkl
#   sbatch slurm/run_diagnostics.sh A results/workflow_A_npe_20260204_230502/npe_model.pkl
#

#SBATCH --job-name=npe_diag
#SBATCH --output=slurm/outputs/npe_diag_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Positional arguments
MODEL="${1:-original}"
MODEL_PATH="${2}"

if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH is required."
    echo "Usage: sbatch slurm/run_diagnostics.sh <MODEL> <MODEL_PATH>"
    exit 1
fi

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
echo "SBI DIAGNOSTICS: model=${MODEL}"
echo "Model path: ${MODEL_PATH}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================"

python src/run_diagnostics.py \
    --model_path "${MODEL_PATH}" \
    --model "${MODEL}" \
    --n_sbc_sims 1000 \
    --n_posterior_samples 1000 \
    --n_workers 8 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
