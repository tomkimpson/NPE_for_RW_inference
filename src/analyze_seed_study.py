#!/usr/bin/env python3
"""
Aggregate seed-study results across 4 models x 5 seeds and produce
a reproducibility summary table (plain text + LaTeX).

Usage:
    python src/analyze_seed_study.py [--results_dir results/seed_study]
                                     [--output_tex results/seed_study/seed_study_table.tex]
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


MODELS = ['original', 'A', 'B', 'C']
SEEDS = [42, 123, 456, 789, 1024]


def load_results(results_dir: Path):
    """Load results.pkl for every model+seed combination."""
    all_results = {}
    missing = []

    for model in MODELS:
        for seed in SEEDS:
            pkl_path = results_dir / model / f'seed_{seed}' / 'inference_results' / 'results.pkl'
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    all_results[(model, seed)] = pickle.load(f)
            else:
                missing.append((model, seed))

    if missing:
        print(f"WARNING: {len(missing)} result files missing:")
        for model, seed in missing:
            print(f"  {model}/seed_{seed}")
        print()

    print(f"Loaded {len(all_results)}/{len(MODELS) * len(SEEDS)} results.\n")
    return all_results


def compute_stats(all_results):
    """Compute per-parameter, per-model statistics across seeds."""
    stats = {}

    for model in MODELS:
        seed_results = {s: all_results[(model, s)]
                        for s in SEEDS if (model, s) in all_results}
        if not seed_results:
            continue

        param_names = list(seed_results.values())[0]['param_names']
        true_params = list(seed_results.values())[0]['true_parameters']
        n_params = len(param_names)

        model_stats = {'param_names': param_names, 'true_parameters': true_params,
                        'n_seeds': len(seed_results), 'params': {}}

        for pidx, pname in enumerate(param_names):
            means = []
            ci_widths = []
            true_in_ci = []

            for seed, res in sorted(seed_results.items()):
                samples = res['posterior_samples'][:, pidx]
                pmean = float(np.mean(samples))
                ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])
                ci_width = ci_hi - ci_lo
                in_ci = ci_lo <= true_params[pidx] <= ci_hi

                means.append(pmean)
                ci_widths.append(ci_width)
                true_in_ci.append(in_ci)

            means = np.array(means)
            ci_widths = np.array(ci_widths)
            true_in_ci = np.array(true_in_ci)

            model_stats['params'][pname] = {
                'true_value': true_params[pidx],
                'posterior_means': means,
                'mean_of_means': float(np.mean(means)),
                'std_of_means': float(np.std(means)),
                'ci_widths': ci_widths,
                'mean_ci_width': float(np.mean(ci_widths)),
                'std_ci_width': float(np.std(ci_widths)),
                'true_in_ci': true_in_ci,
                'coverage': float(np.mean(true_in_ci)),
            }

        stats[model] = model_stats

    return stats


def print_text_table(stats):
    """Print a plain-text summary table."""
    print("=" * 90)
    print("SEED STUDY REPRODUCIBILITY SUMMARY")
    print("=" * 90)

    for model in MODELS:
        if model not in stats:
            continue
        ms = stats[model]
        n = ms['n_seeds']
        print(f"\nModel: {model}  ({n} seeds)")
        print(f"{'Param':<8} {'True':>6} {'Mean':>12} {'Std':>8} "
              f"{'CI width':>14} {'Coverage':>10}")
        print("-" * 70)

        for pname, ps in ms['params'].items():
            print(f"{pname:<8} {ps['true_value']:>6.3f} "
                  f"{ps['mean_of_means']:>7.4f} +/- {ps['std_of_means']:<6.4f} "
                  f"{ps['mean_ci_width']:>7.4f} +/- {ps['std_ci_width']:<6.4f} "
                  f"{ps['coverage']:>7.0%}")

    print()


def generate_latex_table(stats):
    """Generate a LaTeX table string."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reproducibility across random seeds. "
                 r"For each model and parameter we report the mean $\pm$ std of "
                 r"the posterior mean and 95\% CI width across 5 seeds, "
                 r"plus the coverage (fraction of seeds where the true value "
                 r"falls within the 95\% CI).}")
    lines.append(r"\label{tab:seed_study}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Param & True & Post.\ mean & Post.\ std & "
                 r"CI width & Coverage \\")
    lines.append(r"\midrule")

    for i, model in enumerate(MODELS):
        if model not in stats:
            continue
        ms = stats[model]
        params = list(ms['params'].items())

        for j, (pname, ps) in enumerate(params):
            model_col = model if j == 0 else ""
            lines.append(
                f"  {model_col} & ${pname}$ & "
                f"${ps['true_value']:.3f}$ & "
                f"${ps['mean_of_means']:.4f} \\pm {ps['std_of_means']:.4f}$ & "
                f"${ps['std_of_means']:.4f}$ & "
                f"${ps['mean_ci_width']:.4f} \\pm {ps['std_ci_width']:.4f}$ & "
                f"${ps['coverage']:.0%}$ \\\\"
            )

        if i < len(MODELS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze seed study results')
    parser.add_argument('--results_dir', type=str,
                        default='results/seed_study',
                        help='Root directory of seed study results')
    parser.add_argument('--output_tex', type=str, default=None,
                        help='Save LaTeX table to file')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load all results
    all_results = load_results(results_dir)
    if not all_results:
        print("ERROR: No results loaded. Exiting.")
        sys.exit(1)

    # Compute statistics
    stats = compute_stats(all_results)

    # Print plain-text table
    print_text_table(stats)

    # Generate and print LaTeX table
    latex = generate_latex_table(stats)
    print("LaTeX Table:")
    print(latex)

    # Optionally save LaTeX table
    if args.output_tex:
        tex_path = Path(args.output_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, 'w') as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {tex_path}")

    # Save full stats pickle
    stats_path = results_dir / 'seed_study_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Full stats saved to: {stats_path}")


if __name__ == '__main__':
    main()
