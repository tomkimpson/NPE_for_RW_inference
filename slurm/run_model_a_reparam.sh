#!/bin/bash
#
# Model A reparameterized: train NPE in continuum space (U, D, v)
# instead of lattice space (U, P, rho).
#
# Reuses existing 50k training data from Model A 2D run.
# Only theta columns are transformed; observations are unchanged.
#
# Usage:
#   sbatch slurm/run_model_a_reparam.sh
#

#SBATCH --job-name=npe_a_reparam
#SBATCH --output=slurm/outputs/npe_a_reparam_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

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
echo "MODEL A REPARAMETERIZED: (U, D, v)"
echo "  Reusing existing 50k training data"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================"

python src/main.py \
    --model A \
    --reparam_continuum \
    --use_2d_data \
    --disable_sbi_standardization \
    --cnn_spatial_pyramid \
    --skip_data --data_path /fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/results/workflow_A_npe2d_20260206_160159/training_data.pkl \
    --n_samples 50000 \
    --hidden_features 256 --num_transforms 8 \
    --max_epochs 100 --stop_after_epochs 20 \
    --batch_size 512 --learning_rate 1e-4 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 5000 \
    --n_pred_samples 500 \
    --n_workers 8 \
    --seed 42

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
