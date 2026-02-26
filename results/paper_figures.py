#!/usr/bin/env python3
"""
Generate publication-quality figures for the NPE random walk paper.

Produces:
  A. Corner plots from 1D NPE posteriors (one per model)
  B. 1D vs 2D comparison corner plots (overlay on same axes)
  C. Posterior predictive interval plots

Usage:
  python results/paper_figures.py --output-dir results/paper_figures/
  python results/paper_figures.py --output-dir results/paper_figures/ --models A C
  python results/paper_figures.py --output-dir results/paper_figures/ --format png
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import corner

try:
    import scienceplots  # noqa: F401 -- registers styles with matplotlib
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import MODEL_CONFIGS, get_model_config  # noqa: E402


def _apply_style():
    """Apply publication-quality style if scienceplots is available."""
    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "no-latex"])
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        })


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLOR_1D = "#4682B4"      # steel blue
COLOR_2D = "#E67E22"      # orange
COLOR_TRUTH = "#C0392B"   # red
COLOR_PRED_BAND = "#27AE60"  # green

PARAM_COLORS = {
    "U":   "#4682B4",   # steel blue
    "P":   "#E67E22",   # orange
    "rho": "#27AE60",   # green
    "R":   "#8E44AD",   # purple
    "D":   "#E67E22",   # orange (same slot as P)
    "v":   "#27AE60",   # green (same slot as rho)
}

SINGLE_COL_WIDTH_IN = 3.46   # ~88 mm
DOUBLE_COL_WIDTH_IN = 7.09   # ~180 mm
DPI = 300

# Mapping from (model, approach) -> result directory (relative to repo root).
# approach: 'npe' (1D), 'npe2d' (2D), 'abc'
RESULT_PATHS = {
    ("original", "npe"):   "results/workflow_original_npe_20260226_094822",  # 50k sims, 256 hidden features
    ("original", "npe2d"): "results/workflow_original_npe2d_20260223_104550",  # 50k sims, 256 hidden features
    ("original", "abc"):   "results/workflow_original_abc_20260204_225446",
    ("A", "npe"):          "results/workflow_A_npe_20260226_095713",          # 50k sims, 256 hidden features
    ("A", "npe2d"):        "results/workflow_A_npe2d_20260225_140808",        # 50k, SPP + no sbi standardization
    ("B", "npe"):          "results/workflow_B_npe_20260226_110956",          # 50k sims, 256 hidden features
    ("B", "npe2d"):        "results/workflow_B_npe2d_20260224_091648",        # R=0.05, 50k sims
    ("C", "npe"):          "results/workflow_C_npe_20260226_094822",          # 50k sims, 256 hidden features, 4-param
    ("C", "npe2d"):        "results/workflow_C_npe2d_20260224_171232",        # 50k sims, 4-param
    ("A_reparam", "npe2d"): "results/workflow_A_npe2d_20260226_083659",      # reparam (U,D,v), 50k sims
}

# LaTeX-style labels for each parameter
PARAM_LATEX = {
    "U":   r"$U$",
    "P":   r"$P$",
    "rho": r"$\rho$",
    "R":   r"$R$",
    "D":   r"$D$",
    "v":   r"$v$",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(model: str, approach: str):
    """Load inference results (posterior samples + truth) for a model/approach."""
    key = (model, approach)
    if key not in RESULT_PATHS:
        raise FileNotFoundError(f"No result path configured for {key}")
    results_pkl = REPO_ROOT / RESULT_PATHS[key] / "inference_results" / "results.pkl"
    if not results_pkl.exists():
        raise FileNotFoundError(f"Results file not found: {results_pkl}")
    with open(results_pkl, "rb") as f:
        data = pickle.load(f)
    return data


def load_predictive(model: str, approach: str):
    """Load posterior predictive results for a model/approach."""
    key = (model, approach)
    if key not in RESULT_PATHS:
        raise FileNotFoundError(f"No result path configured for {key}")
    pred_pkl = REPO_ROOT / RESULT_PATHS[key] / "inference_results" / "predictive_results.pkl"
    if not pred_pkl.exists():
        raise FileNotFoundError(f"Predictive results file not found: {pred_pkl}")
    with open(pred_pkl, "rb") as f:
        data = pickle.load(f)
    return data


def load_diagnostics(model: str, approach: str = "npe"):
    """Load diagnostics results (SBC ranks, TARP, etc.) for a model/approach."""
    key = (model, approach)
    if key not in RESULT_PATHS:
        raise FileNotFoundError(f"No result path configured for {key}")
    diag_pkl = REPO_ROOT / RESULT_PATHS[key] / "diagnostics" / "diagnostics_results.pkl"
    if not diag_pkl.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {diag_pkl}")
    with open(diag_pkl, "rb") as f:
        data = pickle.load(f)
    return data


def _labels_and_ranges(model: str):
    """Return (labels, ranges) for a model's parameters from ModelConfig."""
    cfg = get_model_config(model)
    labels = [PARAM_LATEX.get(n, n) for n in cfg.param_names]
    ranges = list(zip(cfg.prior_low, cfg.prior_high))
    return labels, ranges


# ---------------------------------------------------------------------------
# Figure A: Corner plots (1D NPE)
# ---------------------------------------------------------------------------

def make_corner_plot(model: str, approach: str, output_dir: Path, fmt: str):
    """Single corner plot from NPE posterior.

    Parameters
    ----------
    model : str
        Model name ('original', 'A', 'B', 'C').
    approach : str
        'npe' for 1D column counts, 'npe2d' for 2D spatial (CNN).
    output_dir : Path
        Directory for output figures.
    fmt : str
        File format ('pdf' or 'png').
    """
    data = load_results(model, approach)
    samples = data["posterior_samples"].copy()
    truths = list(data["true_parameters"])
    labels, ranges = _labels_and_ranges(model)

    # For the original model, reparameterize P -> D = P/4
    if model == "original":
        p_idx = 1  # P is the second parameter
        samples[:, p_idx] = samples[:, p_idx] / 4.0
        truths[p_idx] = truths[p_idx] / 4.0
        labels[p_idx] = r"$D$"
        lo, hi = ranges[p_idx]
        ranges[p_idx] = (lo / 4.0, hi / 4.0)

    color = COLOR_1D if approach == "npe" else COLOR_2D
    suffix = "1d" if approach == "npe" else "2d"

    n_params = len(labels)
    width = SINGLE_COL_WIDTH_IN if n_params <= 2 else DOUBLE_COL_WIDTH_IN
    title_fs = 8 if n_params <= 2 else 11
    label_fs = 9 if n_params <= 2 else 12

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        truth_color=COLOR_TRUTH,
        color=color,
        show_titles=True,
        title_kwargs={"fontsize": title_fs},
        label_kwargs={"fontsize": label_fs},
        title_fmt=".3f",
        range=ranges,
        smooth=1.0,
        smooth1d=1.0,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
    )
    fig.set_size_inches(width, width)

    outfile = output_dir / f"corner_{model}_{suffix}.{fmt}"
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Figure A2: Reparameterization comparison corner plot (side-by-side)
# ---------------------------------------------------------------------------

def make_reparam_corner_comparison(output_dir: Path, fmt: str):
    """Generate two separate corner plots for LaTeX subfigure composition:
    baseline (U,P,rho) and reparameterized (U,D,v) for Model A 2D.

    Outputs:
        corner_A_2d_baseline.{fmt}  -- baseline (U, P, rho)
        corner_A_2d_reparam.{fmt}   -- reparameterized (U, D, v)
    """
    cfg_a = get_model_config("A")

    title_fs = 11
    label_fs = 12
    panel_width = SINGLE_COL_WIDTH_IN * 1.4

    corner_kwargs = dict(
        truth_color=COLOR_TRUTH,
        color=COLOR_2D,
        show_titles=True,
        title_kwargs={"fontsize": title_fs},
        label_kwargs={"fontsize": label_fs},
        title_fmt=".3f",
        smooth=1.0,
        smooth1d=1.0,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
    )

    # --- Left panel: baseline (U, P, rho) ---
    data_base = load_results("A", "npe2d")
    samples_base = data_base["posterior_samples"].copy()
    truths_base = list(data_base["true_parameters"])
    labels_base = [r"$U$", r"$P$", r"$\rho$"]
    ranges_base = list(zip(cfg_a.prior_low, cfg_a.prior_high))

    fig_left = corner.corner(
        samples_base, labels=labels_base, truths=truths_base,
        range=ranges_base, **corner_kwargs,
    )
    fig_left.set_size_inches(panel_width, panel_width)
    outfile_left = output_dir / f"corner_A_2d_baseline.{fmt}"
    fig_left.savefig(outfile_left, dpi=DPI, bbox_inches="tight")
    plt.close(fig_left)
    print(f"  Saved {outfile_left}")

    # --- Right panel: reparameterized (U, D, v) ---
    data_repa = load_results("A_reparam", "npe2d")
    samples_repa = data_repa["posterior_samples"].copy()
    truths_repa = list(data_repa["true_parameters"])
    labels_repa = [PARAM_LATEX.get(n, n) for n in data_repa["param_names"]]

    # Ranges: U same; D = P/4; v = P*rho/2
    u_lo, u_hi = cfg_a.prior_low[0], cfg_a.prior_high[0]
    p_lo, p_hi = cfg_a.prior_low[1], cfg_a.prior_high[1]
    rho_lo, rho_hi = cfg_a.prior_low[2], cfg_a.prior_high[2]
    ranges_repa = [
        (u_lo, u_hi),
        (p_lo / 4.0, p_hi / 4.0),
        (p_lo * rho_lo / 2.0, p_hi * rho_hi / 2.0),
    ]

    fig_right = corner.corner(
        samples_repa, labels=labels_repa, truths=truths_repa,
        range=ranges_repa, **corner_kwargs,
    )
    fig_right.set_size_inches(panel_width, panel_width)
    outfile_right = output_dir / f"corner_A_2d_reparam.{fmt}"
    fig_right.savefig(outfile_right, dpi=DPI, bbox_inches="tight")
    plt.close(fig_right)
    print(f"  Saved {outfile_right}")


# ---------------------------------------------------------------------------
# Figure B: 1D vs 2D comparison corner plots
# ---------------------------------------------------------------------------

def make_comparison_plot(model: str, output_dir: Path, fmt: str):
    """Overlay 1D and 2D posteriors on the same corner plot."""
    data_1d = load_results(model, "npe")
    data_2d = load_results(model, "npe2d")
    truths = data_1d["true_parameters"]
    labels, ranges = _labels_and_ranges(model)

    n_params = len(labels)
    width = DOUBLE_COL_WIDTH_IN

    # Plot 2D first (background)
    fig = corner.corner(
        data_2d["posterior_samples"],
        labels=labels,
        truths=truths,
        truth_color=COLOR_TRUTH,
        color=COLOR_2D,
        show_titles=False,
        label_kwargs={"fontsize": 9},
        range=ranges,
        smooth=1.0,
        smooth1d=1.0,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
    )

    # Overlay 1D (foreground)
    corner.corner(
        data_1d["posterior_samples"],
        labels=labels,
        color=COLOR_1D,
        show_titles=False,
        label_kwargs={"fontsize": 9},
        range=ranges,
        smooth=1.0,
        smooth1d=1.0,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
        fig=fig,
    )

    fig.set_size_inches(width, width)

    # Manual legend
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=COLOR_1D, lw=2, label="1D NPE"),
        mlines.Line2D([], [], color=COLOR_2D, lw=2, label="2D NPE (CNN)"),
        mlines.Line2D([], [], color=COLOR_TRUTH, lw=1.5, ls="--", label="True value"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    outfile = output_dir / f"comparison_{model}.{fmt}"
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Figure C: Prediction interval plots
# ---------------------------------------------------------------------------

def make_prediction_plot(model: str, output_dir: Path, fmt: str):
    """Publication-quality posterior predictive interval plot."""
    pred_data = load_predictive(model, "npe")
    prediction_results = pred_data["prediction_results"]
    observed_data = pred_data.get("observed_data", None)
    intervals = prediction_results["intervals"]

    # Determine Lx from the prediction data
    n_columns = prediction_results["metadata"]["n_columns"]
    x_min = -(n_columns // 2)
    x_max = n_columns // 2 if n_columns % 2 == 1 else (n_columns // 2) - 1
    columns = np.arange(x_min, x_max + 1)

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH_IN, DOUBLE_COL_WIDTH_IN * 0.45))

    # 95% band
    ax.fill_between(
        columns, intervals["p2.5"], intervals["p97.5"],
        alpha=0.2, color=COLOR_PRED_BAND, label="95% prediction interval",
    )
    # 50% band
    ax.fill_between(
        columns, intervals["p25"], intervals["p75"],
        alpha=0.35, color=COLOR_PRED_BAND, label="50% prediction interval",
    )
    # Median line
    ax.plot(columns, intervals["p50"], "-", color=COLOR_PRED_BAND, lw=1.5, label="Median")

    # Observed data
    if observed_data is not None:
        ax.scatter(
            columns, observed_data, c=COLOR_1D, s=12, zorder=5,
            label="Observed data", edgecolors="none",
        )

    ax.set_xlabel("Column index (centred)")
    ax.set_ylabel("Agent count")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    outfile = output_dir / f"prediction_{model}.{fmt}"
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Figure D: SBC rank ECDF plots
# ---------------------------------------------------------------------------

def make_sbc_plot(model: str, output_dir: Path, fmt: str, approach: str = "npe"):
    """Publication-quality SBC rank ECDF plot with 1D and 2D overlaid.

    1D results are plotted as solid lines; 2D results (if available) are
    overlaid as dashed lines in the same colours.

    Parameters
    ----------
    model : str
        Model name ('original', 'A', 'B', 'C').
    output_dir : Path
        Directory for output figures.
    fmt : str
        File format ('pdf' or 'png').
    approach : str
        Kept for backwards compatibility; the function always loads 1D
        and attempts to overlay 2D.
    """
    diag_1d = load_diagnostics(model, "npe")
    ranks_1d = diag_1d["sbc_ranks"]
    if isinstance(ranks_1d, torch.Tensor):
        ranks_1d = ranks_1d.numpy()

    # Try loading 2D diagnostics
    try:
        diag_2d = load_diagnostics(model, "npe2d")
        ranks_2d = diag_2d["sbc_ranks"]
        if isinstance(ranks_2d, torch.Tensor):
            ranks_2d = ranks_2d.numpy()
        has_2d = True
    except FileNotFoundError:
        has_2d = False

    n_sbc, n_params = ranks_1d.shape
    ranks_1d_norm = ranks_1d / ranks_1d.max()

    if has_2d:
        ranks_2d_norm = ranks_2d / ranks_2d.max()
        n_sbc_2d = ranks_2d.shape[0]

    cfg = get_model_config(model)
    param_names = cfg.param_names

    fig_h = 0.8 * SINGLE_COL_WIDTH_IN
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL_WIDTH_IN, fig_h))

    # Uniform reference diagonal
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=0.8, zorder=1)

    # DKW 99% confidence band (use 1D n_sbc)
    alpha = 0.01
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * n_sbc))
    t = np.linspace(0, 1, 200)
    ax.fill_between(
        t,
        np.clip(t - epsilon, 0, 1),
        np.clip(t + epsilon, 0, 1),
        color="0.85", zorder=0, label="99% DKW band",
    )

    # Plot 1D ECDFs (solid lines)
    ecdf_y_1d = np.arange(1, n_sbc + 1) / n_sbc
    for j in range(n_params):
        name = param_names[j]
        sorted_ranks = np.sort(ranks_1d_norm[:, j])
        label = PARAM_LATEX.get(name, name) + " (1D)"
        color = PARAM_COLORS.get(name, f"C{j}")
        ax.plot(sorted_ranks, ecdf_y_1d, color=color, lw=1.2, ls="-",
                label=label, zorder=2)

    # Overlay 2D ECDFs (dashed lines)
    if has_2d:
        ecdf_y_2d = np.arange(1, n_sbc_2d + 1) / n_sbc_2d
        for j in range(n_params):
            name = param_names[j]
            sorted_ranks = np.sort(ranks_2d_norm[:, j])
            label = PARAM_LATEX.get(name, name) + " (2D)"
            color = PARAM_COLORS.get(name, f"C{j}")
            ax.plot(sorted_ranks, ecdf_y_2d, color=color, lw=1.2, ls="--",
                    label=label, zorder=2)

    ax.set_xlabel("Normalized rank")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="upper left", ncol=2 if has_2d else 1)

    tag = model if model == "original" else f"model_{model}"
    outfile = output_dir / f"sbc_{tag}.{fmt}"
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from NPE results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/paper_figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--models", nargs="+", default=["original", "A", "B", "C"],
        help="Which models to generate figures for",
    )
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png"],
        help="Output file format",
    )
    args = parser.parse_args()

    # Apply publication-quality style
    _apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    print(f"Output directory: {output_dir}")
    print(f"Format: {fmt}")
    print(f"Models: {args.models}\n")

    for model in args.models:
        print(f"--- Model {model} ---")

        # Corner plots for both 1D and 2D
        for approach in ["npe", "npe2d"]:
            try:
                make_corner_plot(model, approach, output_dir, fmt)
            except FileNotFoundError as e:
                print(f"  SKIP corner {approach}: {e}")

        # Prediction intervals
        try:
            make_prediction_plot(model, output_dir, fmt)
        except FileNotFoundError as e:
            print(f"  SKIP prediction plot: {e}")

        # SBC rank ECDF (1D + 2D overlay)
        try:
            make_sbc_plot(model, output_dir, fmt)
        except FileNotFoundError as e:
            print(f"  SKIP SBC plot: {e}")

        print()

    # Reparameterization comparison (Model A only)
    if "A" in args.models:
        print("--- Model A reparameterization comparison ---")
        try:
            make_reparam_corner_comparison(output_dir, fmt)
        except FileNotFoundError as e:
            print(f"  SKIP reparam comparison: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
