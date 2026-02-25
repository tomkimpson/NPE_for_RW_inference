#!/bin/bash
#
# Seed study: single model+seed run with 50k samples for fair 1D/2D comparison.
# Uses pre-converted 1D training data from results/training_data_1d_50k/.
#
# Usage:  sbatch slurm/run_seed_single_50k.sh <MODEL> <SEED>
#
# Examples:
#   sbatch slurm/run_seed_single_50k.sh original 42
#   sbatch slurm/run_seed_single_50k.sh A 123
#

#SBATCH --job-name=npe_seed50k
#SBATCH --output=slurm/outputs/npe_seed50k_%j.txt
#SBATCH --export=ALL
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Positional arguments
MODEL="${1:-original}"
SEED="${2:-42}"

# Fixed observation seed — matches existing production runs
OBS_SEED=1042

# Pre-converted 1D training data
DATA_PATH="results/training_data_1d_50k/${MODEL}_training_data_1d.pkl"

# Output directory for this model+seed combo
OUTPUT_DIR="results/seed_study_50k/${MODEL}/seed_${SEED}"

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
echo "SEED STUDY (50k): model=${MODEL}  seed=${SEED}  obs_seed=${OBS_SEED}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================"

python src/main.py \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --obs_seed "${OBS_SEED}" \
    --n_samples 50000 \
    --max_epochs 100 --stop_after_epochs 20 \
    --hidden_features 256 --num_transforms 8 \
    --learning_rate 1e-4 \
    --batch_size 512 \
    --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25 \
    --num_samples 5000 \
    --n_pred_samples 500 \
    --n_workers 16 \
    --skip_data --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
