# Notebooks

These notebooks are built with [marimo](https://marimo.io), a reactive Python notebook framework. They are stored as plain `.py` files and can be launched interactively or run as scripts.

## Launching

```bash
# Install marimo
pip install marimo

# Open a notebook in the browser (editable)
marimo edit notebooks/demo.py

# Or run non-interactively
marimo run notebooks/demo.py

# Or execute as a plain Python script
python notebooks/demo.py
```

## Contents

| Notebook | Description |
|----------|-------------|
| `demo.py` | Interactive tutorial walking through the full NPE workflow |
| `pde_comparison_figure.py` | Generates the PDE vs stochastic simulation comparison figure (paper Figure 1) |
| `reproduce_ABC_results.py` | Reproduces classical ABC and MCMC baselines from Simpson & Plank (2025) |
| `reproduce_ABC_results_just_surrogate.py` | Surrogate-only variant of the classical baselines |

A Jupyter-compatible export of the demo is also available at `__marimo__/demo.ipynb`.
