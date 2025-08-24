#SBATCH --job-name=npe_random_walk
#SBATCH --output=slurm/outputs/npe_random_walk_%j.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create output directory if it doesn't exist
mkdir -p slurm/outputs

source ~/.bashrc
source venv/bin/activate
cd /fred/oz022/tkimpson/NPE_for_RW_inference
time python -u src/main.py --Lx 100 --Ly 50 --T 100 --n_samples 1000
