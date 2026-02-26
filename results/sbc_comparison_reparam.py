#!/usr/bin/env python3
"""
Generate SBC rank ECDF comparison plot: baseline Model A 2D (U,P,rho) vs
reparameterized Model A (U,D,v).

Usage:
    python results/sbc_comparison_reparam.py
"""

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIAG = Path("/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/results/"
                     "workflow_A_npe2d_20260225_140808/diagnostics/diagnostics_results.pkl")
REPARAM_DIAG = REPO_ROOT / "results/workflow_A_npe2d_20260226_083659/diagnostics/diagnostics_results.pkl"
OUTPUT_DIR = REPO_ROOT / "results" / "figures"

# Style
SINGLE_COL_WIDTH_IN = 3.46
DPI = 300

BASELINE_PARAMS = ["U", "P", r"$\rho$"]
REPARAM_PARAMS = ["U", "D", "v"]

COLORS_BASELINE = {"U": "#4682B4", "P": "#E67E22", r"$\rho$": "#27AE60"}
COLORS_REPARAM = {"U": "#4682B4", "D": "#E67E22", "v": "#27AE60"}


def load_ranks(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    ranks = data["sbc_ranks"]
    if isinstance(ranks, torch.Tensor):
        ranks = ranks.numpy()
    return ranks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ranks_base = load_ranks(BASELINE_DIAG)
    ranks_repa = load_ranks(REPARAM_DIAG)

    n_sbc_base = ranks_base.shape[0]
    n_sbc_repa = ranks_repa.shape[0]

    # Normalize ranks to [0, 1]
    ranks_base_norm = ranks_base / ranks_base.max()
    ranks_repa_norm = ranks_repa / ranks_repa.max()

    # --- Two-panel figure: baseline (left), reparam (right) ---
    fig, (ax_base, ax_repa) = plt.subplots(
        1, 2, figsize=(SINGLE_COL_WIDTH_IN * 2, SINGLE_COL_WIDTH_IN * 0.85),
        sharey=True,
    )

    for ax, ranks_norm, n_sbc, param_labels, colors, title in [
        (ax_base, ranks_base_norm, n_sbc_base, BASELINE_PARAMS, COLORS_BASELINE,
         r"Baseline $(U, P, \rho)$"),
        (ax_repa, ranks_repa_norm, n_sbc_repa, REPARAM_PARAMS, COLORS_REPARAM,
         r"Reparameterized $(U, D, v)$"),
    ]:
        # Diagonal reference
        ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=0.8, zorder=1)

        # DKW 99% confidence band
        alpha = 0.01
        epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * n_sbc))
        t = np.linspace(0, 1, 200)
        ax.fill_between(
            t,
            np.clip(t - epsilon, 0, 1),
            np.clip(t + epsilon, 0, 1),
            color="0.85", zorder=0, label="99% DKW band",
        )

        # ECDFs
        ecdf_y = np.arange(1, n_sbc + 1) / n_sbc
        n_params = ranks_norm.shape[1]
        for j in range(n_params):
            name = param_labels[j]
            sorted_ranks = np.sort(ranks_norm[:, j])
            color = colors[name]
            ax.plot(sorted_ranks, ecdf_y, color=color, lw=1.4, ls="-",
                    label=f"${name}$" if not name.startswith("$") else name,
                    zorder=2)

        ax.set_xlabel("Normalized rank", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    ax_base.set_ylabel("ECDF", fontsize=8)

    fig.tight_layout(w_pad=1.5)
    outfile = OUTPUT_DIR / "sbc_model_A_reparam_comparison.png"
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
