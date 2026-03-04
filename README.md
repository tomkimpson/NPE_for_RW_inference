# NPE for Random Walk Inference

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.10.26.684706-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.10.26.684706v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository implements Neural Posterior Estimation (NPE) for inferring the parameters of stochastic random walk models of barrier assay experiments, as described in our paper (Kimpson, Flegg & Simpson, submitted to JTB).

We consider four model variants of increasing complexity:

| Model | Parameters inferred | Mechanisms |
|-------|-------------------|------------|
| **Original** | U, P | Migration only (no exclusion) |
| **Model A** | U, P, rho | Exclusion + directional bias |
| **Model B** | U, P, R | Exclusion + proliferation |
| **Model C** | U, P, rho, R | Exclusion + bias + proliferation |

where **U** is the initial occupancy probability, **P** is the movement probability, **rho** is a directional bias parameter, and **R** is the proliferation rate.

The NPE approach supports both **1D** (column-count summary statistics) and **2D** (full spatial grid via CNN embedding) data representations.

## Installation

### Requirements
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

```bash
git clone https://github.com/tomkimpson/NPE_for_RW_inference.git
cd NPE_for_RW_inference

# Create conda environment (recommended)
conda create -n npe_rw python=3.12
conda activate npe_rw

# Install dependencies
pip install -r requirements.txt
```

**For GPU support** (optional):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Interactive tutorial

The best way to get started is the interactive [marimo](https://marimo.io) notebook:

```bash
marimo edit notebooks/demo.py
```

This walks through the full workflow — data generation, training, and inference — with explanations.

### Command-line usage

```bash
# Run with default settings (original model, 1D data)
python src/main.py

# Choose a model variant
python src/main.py --model A
python src/main.py --model C --n_samples 50000

# Use 2D spatial data with CNN embedding
python src/main.py --model A --use_2d_data

# Custom training parameters
python src/main.py --model B --n_samples 50000 --max_epochs 100 --hidden_features 256

# Reuse existing training data (skip regeneration)
python src/main.py --model A --skip_data --data_path results/workflow_.../training_data.pkl

# Force CPU/GPU
python src/main.py --device cpu
python src/main.py --device cuda
```

### Posterior predictive sampling

After running the main workflow:

```bash
python src/predict.py results/workflow_.../inference_results/results.pkl
```

### SBI diagnostics (SBC, C2ST, TARP)

```bash
python src/run_diagnostics.py results/workflow_.../
```

## Project Structure

```
src/
├── main.py              # Entry point — data generation, training, inference
├── simulator.py         # Random walk simulators (standard + exclusion process)
├── inference.py         # NPE training and posterior sampling
├── models.py            # Model configurations (original, A, B, C)
├── cnn_utils.py         # CNN embedding network for 2D spatial data
├── predict.py           # Posterior predictive sampling
├── diagnostics.py       # SBC, C2ST, TARP diagnostics
├── run_diagnostics.py   # Diagnostics entry point
├── abc_inference.py     # ABC-SMC baseline
└── utils.py             # Posterior statistics and helpers

notebooks/
├── demo.py              # Interactive tutorial (marimo notebook)
├── NPE_results.py       # Results analysis
├── reproduce_ABC_results.py          # Classical baseline reproduction
├── reproduce_ABC_results_just_surrogate.py
└── pde_comparison_figure.py          # Paper figure generation

docs/paper/              # Manuscript, figures, and LaTeX source
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `original` | Model variant: `original`, `A`, `B`, or `C` |
| `--use_2d_data` | off | Use full 2D spatial grid (CNN embedding) instead of 1D column counts |
| `--n_samples` | 10000 | Number of training simulations |
| `--max_epochs` | 100 | Maximum training epochs |
| `--hidden_features` | 128 | Neural spline flow width |
| `--num_transforms` | 5 | Number of coupling transforms |
| `--Lx`, `--Ly` | 200, 50 | Lattice dimensions |
| `--T` | 100 | Simulation time steps |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

## Output

Each run creates a timestamped directory in `results/`:

```
results/workflow_YYYYMMDD_HHMMSS/
├── config.txt               # Run configuration
├── training_data.pkl        # Training simulations
├── npe_model.pkl            # Trained NPE model
├── inference_results/       # Posterior samples, plots, summary
└── predictions/             # Posterior predictive results (optional)
```

## Citation

If you use this code, please cite:

```bibtex
@article{kimpson2025npe,
  author  = {Kimpson, Tom and Flegg, Mark B. and Simpson, Matthew J.},
  title   = {Rapid parameter inference for stochastic models of biological
             populations undergoing migration and proliferation using
             neural posterior estimation},
  journal = {Journal of Theoretical Biology},
  year    = {2025},
  note    = {Submitted}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
