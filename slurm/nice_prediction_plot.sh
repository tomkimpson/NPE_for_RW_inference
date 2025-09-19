#!/bin/bash

#SBATCH --job-name=nice_prediction_plot
#SBATCH --output=slurm/outputs/nice_prediction_plot_%j.txt
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

# Activate conda env
source ~/.bashrc
conda activate NPE_LV

# Change to project root directory
cd /fred/oz022/tkimpson/SNPE/NPE_for_RW_inference

# Add src directory to Python path so cnn_utils can be found
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Run nice_prediction_plot.py
time python results/nice_prediction_plot.py results/workflow_20250918_225411/inference_results/posterior_samples.pkl