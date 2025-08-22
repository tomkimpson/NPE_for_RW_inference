# NPE for Random Walk Inference

## Overview

This repository implements Neural Posterior Estimation (NPE) for inferring microscopic parameters of a stochastic random walk model in biological populations, based on the work by [Simpson & Planck](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4).

The goal is to infer two key parameters from observation data:
- **U**: Initial occupancy probability (probability that a site contains an agent at t=0)  
- **P**: Movement probability (probability that an agent moves during a time step)

## Repository Structure

```
NPE_for_RW_Inference/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── run_workflow.py             # Main workflow runner script
├── test_implementation.py      # Test script to validate implementation
├── src/                        # Main source code
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Main workflow script
│   ├── simulator.py           # Random walk simulator
│   ├── inference.py           # NPE training and inference
│   └── utils.py               # Utility functions
├── data/                      # Data files (created during runs)
├── results/                   # Analysis results and outputs
├── docs/                      # Documentation
│   └── summary.md            # Problem description
└── external_code/            # Reference implementations
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
   python test_implementation.py
   ```

## Quick Start

### Run the complete workflow:
```bash
# Basic run with default parameters (auto-detects GPU)
python run_workflow.py

# Custom parameters
python run_workflow.py --n_samples 5000 --max_epochs 50 --theta_true 0.4 0.8

# Force CPU usage
python run_workflow.py --device cpu --n_samples 10000

# Force NVIDIA GPU usage (CUDA)
python run_workflow.py --device cuda --n_samples 10000
```

### Run from src directory:
```bash
cd src
python main.py --help  # See all available options
```

## Workflow Steps

The complete NPE pipeline consists of three main steps:

1. **Training Data Generation**: Generate parameter-observation pairs by running the forward simulation
2. **NPE Training**: Train a neural density estimator to learn the posterior p(U,P|data)  
3. **Inference**: Use the trained model to infer parameters from observed data

## Key Features

- **Flexible simulator**: 2D lattice random walk with configurable dimensions and boundary conditions
- **Neural Posterior Estimation**: Using the `sbi` library with Neural Spline Flows
- **Smart device detection**: Automatically uses CUDA if available, falls back to CPU
- **Data persistence**: Save/load training data as pickle files to avoid regeneration
- **Comprehensive validation**: Coverage assessment and posterior visualization
- **Modular design**: Easy to extend and modify individual components

## Usage Examples

### Basic workflow:
```bash
# Generate 10k training samples, train for 100 epochs, infer parameters
python run_workflow.py --n_samples 10000 --max_epochs 100
```

### Resume from existing data:
```bash
# Skip data generation, use existing training data
python run_workflow.py --skip_data --data_path results/previous_run/training_data.pkl
```

### Test different lattice sizes:
```bash
# Larger lattice, more time steps
python run_workflow.py --Lx 31 --Ly 31 --T 200
```

### Save and reuse training data:
```bash
# First run: generate and save data
python run_workflow.py --save_data my_training_data

# Subsequent runs: reuse the data  
python run_workflow.py --data_path my_training_data
```

*Note: Training data is always saved as pickle (.pkl) files. The .pkl extension is added automatically if not specified.*

## Output Structure

Each workflow run creates a timestamped directory in `results/` containing:

```
results/workflow_YYYYMMDD_HHMMSS/
├── config.txt                    # Run configuration
├── training_data.pkl             # Generated training data
├── npe_model.pkl                 # Trained NPE model
├── inference_results/            # Inference outputs
│   ├── posterior_marginals.png   # Marginal posterior plots
│   ├── posterior_pairwise.png    # Joint posterior visualization
│   ├── observed_data.png         # Observed column counts
│   ├── simulation_comparison.png # Initial vs final states
│   └── results.pkl               # Numerical results
```

## Key Parameters

**Simulation Parameters:**
- `--Lx`, `--Ly`: Lattice dimensions (default: 21×21)
- `--T`: Number of simulation time steps (default: 100)
- `--theta_true`: True parameter values for testing [U, P] (default: [0.3, 0.7])

**Training Parameters:**
- `--n_samples`: Number of training simulations (default: 10,000)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--batch_size`: Training batch size (default: 512)
- `--hidden_features`: Neural network width (default: 128)
- `--device`: Device for training ('cpu', 'cuda', or 'auto', default: 'auto')

## External Code

The `external_code/RandomWalkInference/` directory contains reference implementations from [Simpson and Plank's repository](https://github.com/ProfMJSimpson/RandomWalkInference) for comparison and validation.

## Citation

If you use this code in your research, please cite the original paper:

> Simpson, M.J., & Planck, P. (2025). [Paper title]. bioRxiv. DOI: 10.1101/2025.05.25.656057v4

## License

This project's code is provided for research purposes. The external code in `external_code/` retains its original license from the source repository.