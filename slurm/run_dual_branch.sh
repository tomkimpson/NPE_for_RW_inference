#!/bin/bash
#SBATCH --job-name=npe_dual_branch
#SBATCH --output=slurm/outputs/npe_dual_branch_%j.txt
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

# Dual-branch CNN test for unbiased 2D NPE
# Uses existing training data - only architecture changed

cd /fred/oz022/tkimpson/SNPE/NPE_for_RW_inference

# Activate environment
source ~/.bashrc
conda activate NPE_LV

echo "Starting dual-branch CNN test at $(date)"
echo "Job ID: $SLURM_JOB_ID"

python src/main.py \
    --model A \
    --use_2d_data \
    --cnn_dual_branch \
    --disable_sbi_standardization \
    --skip_data \
    --data_path results/workflow_A_npe2d_20260206_160159/training_data.pkl \
    --n_samples 50000 \
    --max_epochs 100 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --num_samples 1000 \
    --seed 42 \
    --Lx 200 \
    --Ly 50 \
    --T 100

echo "Completed at $(date)"
