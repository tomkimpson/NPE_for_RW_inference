#!/bin/bash 

#SBATCH --job-name=snpe_random_walk
#SBATCH --output=slurm/outputs/snpe_random_walk_%j.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

#Activate conda env
source ~/.bashrc
conda activate NPE_LV 

#Run command - may need to play with the parameters to see what works best. The below are somewhat arbitrary.
time python src/main.py --use_snpe --snpe_rounds 2 --Lx 200 --Ly 50 --T 100 --initial_region_half_width 25