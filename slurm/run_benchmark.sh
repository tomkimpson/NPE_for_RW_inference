#!/bin/bash
#
# Benchmark parallel vs sequential data generation.
# Usage:  sbatch slurm/run_benchmark.sh <MODEL> <N_WORKERS>
# where MODEL is one of: A, B, C
# and   N_WORKERS is the number of parallel workers (1 = sequential)
#

#SBATCH --job-name=npe_bench
#SBATCH --output=slurm/outputs/npe_bench_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Model name and worker count from positional arguments
MODEL="${1:-A}"
N_WORKERS="${2:-1}"

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
echo "BENCHMARK: model=${MODEL}  n_workers=${N_WORKERS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================"

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
    --n_workers "${N_WORKERS}"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
