"""
Generate overlay corner plot comparing 1D column count and 2D spatial posteriors.

Produces Figure 6 for the paper: both posteriors on the same axes with
distinct colors and a legend for direct visual comparison.
"""

import pickle
from pathlib import Path

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 – registers styles


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_1D = Path(
    "/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/results/"
    "workflow_original_npe_20260204_225605/inference_results/results.pkl"
)
RESULTS_2D = Path(
    "/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/results/"
    "workflow_original_npe2d_20260204_230541/inference_results/results.pkl"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "paper" / "images"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_1D = "#2E86C1"  # blue – matches existing Figure 4
COLOR_2D = "#E74C3C"  # red  – clearly distinct
TRUTH_COLOR = "orange"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_reparameterize(path):
    """Load results.pkl and convert P → D = P/4."""
    with open(path, "rb") as f:
        results = pickle.load(f)
    samples = np.array(results["posterior_samples"])   # (N, 2)  columns: [U, P]
    truths = np.array(results["true_parameters"])      # [U_true, P_true]
    # Reparameterize second column: P → D = P/4
    samples[:, 1] = samples[:, 1] / 4.0
    truths[1] = truths[1] / 4.0
    return samples, truths


def compute_ranges(*sample_arrays):
    """Compute axis ranges from the union of all sample arrays with padding."""
    all_samples = np.vstack(sample_arrays)
    lo = all_samples.min(axis=0)
    hi = all_samples.max(axis=0)
    pad = 0.05 * (hi - lo)
    return list(zip(lo - pad, hi + pad))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load data ---
    samples_1d, truths_1d = load_and_reparameterize(RESULTS_1D)
    samples_2d, truths_2d = load_and_reparameterize(RESULTS_2D)

    print(f"1D samples: {samples_1d.shape}, truths: {truths_1d}")
    print(f"2D samples: {samples_2d.shape}, truths: {truths_2d}")

    # --- Styling (matches create_professional_corner_plot) ---
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })

    param_names = [r"$U$", r"$D$"]
    ranges = compute_ranges(samples_1d, samples_2d)
    nbins = 30

    # Shared kwargs for both calls
    common = dict(
        labels=param_names,
        bins=nbins,
        range=ranges,
        show_titles=False,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        max_n_ticks=4,
        use_math_text=True,
        quantiles=[0.16, 0.5, 0.84],
        label_kwargs={"fontsize": 16, "fontweight": "bold"},
    )

    # --- First call: 1D posterior (blue) with truth lines ---
    contour_blues = ["#AED6F1", "#5DADE2", "#2E86C1"]
    fig = corner.corner(
        samples_1d,
        color=COLOR_1D,
        truths=truths_1d,
        truth_color=TRUTH_COLOR,
        truth_kwargs={"linewidth": 2.5, "alpha": 0.8, "linestyle": "--"},
        contour_kwargs={"colors": contour_blues, "linewidths": 0.5},
        hist_kwargs={"alpha": 0.5, "edgecolor": COLOR_1D, "linewidth": 1.0},
        **common,
    )

    # --- Second call: 2D posterior (red), overlaid on same figure ---
    contour_reds = ["#F5B7B1", "#EC7063", "#E74C3C"]
    corner.corner(
        samples_2d,
        fig=fig,
        color=COLOR_2D,
        # No truths on second call – avoids duplicate lines
        contour_kwargs={"colors": contour_reds, "linewidths": 0.5},
        hist_kwargs={"alpha": 0.5, "edgecolor": COLOR_2D, "linewidth": 1.0},
        **common,
    )

    # --- Post-processing (tick styling, grid) ---
    for ax in fig.get_axes():
        if ax is not None:
            ax.tick_params(
                which="major", labelsize=12, width=1.2, length=6,
                direction="in", top=True, right=True,
            )
            ax.tick_params(
                which="minor", width=0.8, length=3,
                direction="in", top=True, right=True,
            )
            ax.minorticks_on()
            ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")

    # --- Legend ---
    handle_1d = mlines.Line2D([], [], color=COLOR_1D, linewidth=2, label="1D column counts")
    handle_2d = mlines.Line2D([], [], color=COLOR_2D, linewidth=2, label="2D spatial (CNN)")
    handle_truth = mlines.Line2D(
        [], [], color=TRUTH_COLOR, linewidth=2, linestyle="--",
        alpha=0.8, label="True values",
    )
    fig.legend(
        handles=[handle_1d, handle_2d, handle_truth],
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.9,
    )

    # --- Layout ---
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "NPE_corner_plot_overlay_1D_2D"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved:\n  {stem.with_suffix('.png')}\n  {stem.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
