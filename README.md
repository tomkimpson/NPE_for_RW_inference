# NPE for Random Walk Inference

## Overview

This repository implements Neural Posterior Estimation (NPE) for inferring the parameters of a stochastic random walk model of barrier assay experiments, based on the work by [Simpson & Planck](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4).

The goal is to infer two key parameters from observation data:
- **U**: Initial occupancy probability (probability that a site contains an agent at t=0)  
- **P**: Movement probability (probability that an agent moves during a time step)

## Repository Structure

```
NPE_for_RW_Inference/
├── README.md                   # This file
├── src/                        # Main source code
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Main workflow script (entry point)
│   ├── simulator.py           # Random walk simulator
│   ├── inference.py           # NPE/SNPE training and inference
│   └── utils.py               # Utility functions
├── results/                   # Analysis results and outputs
├── docs/                      # Documentation
│   └── summary.md            # Problem description
├── slurm/                     # HPC job scripts
│   └── run_main.sh           # SLURM job script for GPU deployment
├── test/                      # Testing and validation
└── notebooks/                # Development notebooks
```

## Installation

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd NPE_for_RW_Inference
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python src/main.py --help
   ```

## Quick Start

### Run the complete workflow:
```bash
# Basic run with default parameters (auto-detects GPU)
python src/main.py

# Custom parameters
python src/main.py --n_samples 5000 --max_epochs 50 --theta_true 0.4 0.8

# Force CPU usage
python src/main.py --device cpu --n_samples 10000

# Force NVIDIA GPU usage (CUDA)
python src/main.py --device cuda --n_samples 10000
```

### Sequential NPE (SNPE) - Recommended for better inference:
```bash
# Basic SNPE run
python src/main.py --use_snpe --snpe_rounds 5

# Advanced SNPE with custom parameters
python src/main.py --use_snpe --snpe_rounds 10 --samples_per_round 2000 \
  --convergence_threshold 0.001 --max_epochs 300
```

### Get help:
```bash
python src/main.py --help  # See all available options
```

## Workflow Steps

The complete pipeline supports two training approaches:

**Standard NPE (3 steps):**
1. **Training Data Generation**: Generate parameter-observation pairs by running the forward simulation
2. **NPE Training**: Train a neural density estimator to learn the posterior p(U,P|data)  
3. **Inference**: Use the trained model to infer parameters from observed data

**Sequential NPE (SNPE) - Recommended:**
1. **Initial Round**: Train on simulations from prior, get initial posterior estimate
2. **Sequential Rounds**: Iteratively refine by simulating from previous posterior, retraining
3. **Convergence**: Stop when posterior estimates stabilize or max rounds reached
4. **Inference**: Use final trained model for parameter inference

## Key Features

- **Dual Training Modes**: Standard NPE and Sequential NPE (SNPE) for improved inference
- **Flexible simulator**: 2D lattice random walk with configurable dimensions and boundary conditions
- **Neural Posterior Estimation**: Using the `sbi` library with Neural Spline Flows
- **Smart device detection**: Automatically uses CUDA if available, falls back to CPU
- **Sequential convergence**: SNPE with automatic convergence detection and early stopping
- **Comprehensive logging**: Round-by-round tracking for SNPE with intermediate results
- **Data persistence**: Save/load training data and models as pickle files
- **Comprehensive validation**: Coverage assessment and posterior visualization
- **HPC ready**: SLURM job scripts included for cluster computing
- **Modular design**: Easy to extend and modify individual components

## Usage Examples

### Standard NPE workflow:
```bash
# Generate 10k training samples, train for 100 epochs, infer parameters
python src/main.py --n_samples 10000 --max_epochs 100
```

### Sequential NPE (SNPE) workflows:
```bash
# Basic SNPE - 5 rounds with auto-determined samples per round
python src/main.py --use_snpe --snpe_rounds 5

# Advanced SNPE with custom configuration
python src/main.py --use_snpe --snpe_rounds 10 --samples_per_round 2000 \
  --convergence_threshold 0.001 --max_epochs 300 --stop_after_epochs 80

# SNPE with larger neural networks (for complex problems)
python src/main.py --use_snpe --hidden_features 768 --num_transforms 15 \
  --learning_rate 5e-6 --batch_size 128
```

### Resume from existing models:
```bash
# Skip training, use existing model for inference
python src/main.py --skip_training --model_path results/previous_run/npe_model.pkl
```

### Test different lattice sizes:
```bash
# Larger lattice, more time steps
python src/main.py --Lx 100 --Ly 50 --T 200

# SNPE on large lattice (recommended for better scaling)
python src/main.py --use_snpe --Lx 100 --Ly 50 --T 200 --snpe_rounds 8
```

### HPC/Cluster usage:
```bash
# Submit SNPE job to SLURM cluster
sbatch slurm/run_main.sh
```

## Output Structure

Each workflow run creates a timestamped directory in `results/` containing:

**Standard NPE Output:**
```
results/workflow_YYYYMMDD_HHMMSS/
├── config.txt                    # Run configuration
├── training_data.pkl             # Generated training data (NPE only)
├── npe_model.pkl                 # Trained NPE model
├── inference_results/            # Inference outputs
│   ├── posterior_marginals.png   # Marginal posterior plots
│   ├── posterior_pairwise.png    # Joint posterior visualization
│   ├── observed_data.png         # Observed column counts
│   ├── simulation_comparison.png # Initial vs final states
│   └── results.pkl               # Numerical results
```

**Sequential NPE (SNPE) Output:**
```
results/workflow_YYYYMMDD_HHMMSS/
├── config.txt                    # Run configuration  
├── npe_model.pkl                 # Final trained SNPE model
├── round_1/                      # First round results
│   ├── posterior_samples.npy     # Posterior samples from round 1
│   └── training_info.pkl         # Round 1 training metadata
├── round_2/                      # Second round results
│   └── ...
├── round_N/                      # Final round results
│   └── ...
├── inference_results/            # Final inference outputs
│   ├── posterior_marginals.png   # Final marginal posterior plots
│   ├── posterior_pairwise.png    # Final joint posterior visualization
│   ├── observed_data.png         # Observed column counts
│   ├── simulation_comparison.png # Initial vs final states
│   └── results.pkl               # Numerical results + SNPE metadata
```

## Key Parameters

**Simulation Parameters:**
- `--Lx`, `--Ly`: Lattice dimensions (default: 21×21)
- `--T`: Number of simulation time steps (default: 100)
- `--theta_true`: True parameter values for testing [U, P] (default: [0.3, 0.7])

**Training Parameters:**
- `--n_samples`: Number of training simulations for NPE (default: 10,000)
- `--max_epochs`: Maximum training epochs per round (default: 100)
- `--batch_size`: Training batch size (default: 512)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--hidden_features`: Neural network width (default: 128)
- `--num_transforms`: Number of coupling transforms (default: 5)
- `--validation_fraction`: Validation data fraction (default: 0.1)
- `--stop_after_epochs`: Early stopping patience (default: 20)
- `--device`: Device for training ('cpu', 'cuda', or 'auto', default: 'auto')

**Sequential NPE (SNPE) Parameters:**
- `--use_snpe`: Enable Sequential Neural Posterior Estimation
- `--snpe_rounds`: Number of sequential rounds (default: 3)
- `--samples_per_round`: Simulations per SNPE round (default: n_samples // snpe_rounds)
- `--convergence_threshold`: Convergence threshold for early stopping (default: 0.01)

## External Code

The `external_code/RandomWalkInference/` directory contains reference implementations from [Simpson and Plank's repository](https://github.com/ProfMJSimpson/RandomWalkInference) for comparison and validation.

## Training Method Comparison

**When to use Standard NPE:**
- Quick prototyping and testing
- Smaller lattice sizes (≤ 31×31)
- Limited computational resources
- When you have good prior knowledge of parameter ranges

**When to use Sequential NPE (SNPE) - Recommended:**
- Better inference accuracy and convergence
- Larger lattice sizes (≥ 50×50)  
- Complex parameter relationships
- Production runs and final results
- When computational resources allow for multiple rounds

SNPE typically achieves better posterior estimates by iteratively refining the training data distribution, focusing simulations on parameter regions more likely given the observed data.

## Citation

If you use this code in your research, please cite the original paper:

> Simpson, M.J., & Planck, P. (2025). [Paper title]. bioRxiv. DOI: 10.1101/2025.05.25.656057v4

## License

This project's code is provided for research purposes. The external code in `external_code/` retains its original license from the source repository.