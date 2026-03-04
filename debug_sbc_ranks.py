"""
Debug script to analyze SBC rank distributions for Model A 2D.

Loads diagnostics_results.pkl, computes rank statistics, and generates
diagnostic histograms to understand the nature of SBC failures for U and rho.

Interpretation of SBC ranks (from Talts et al. 2018):
  rank[i,j] = number of posterior samples where param j < true value
  - If posterior is well-calibrated: ranks ~ Uniform(0, num_posterior_samples)
  - If ranks skew HIGH (hump on right): posterior is biased LOW (too many samples below true)
    => the posterior places too much mass below the true value
  - If ranks skew LOW (hump on left): posterior is biased HIGH (too many samples above true)
    => the posterior places too much mass above the true value
  - If ranks are U-shaped (excess at both extremes): posterior is too narrow (overconfident)
  - If ranks are inverse-U-shaped (hump in middle): posterior is too wide (underconfident)

SBC rank ECDF interpretation:
  - Bowing BELOW the diagonal = ranks are too uniform/spread to extremes = overconfident
  - Bowing ABOVE the diagonal = ranks pile up in the middle = underconfident
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ===========================================================================
# Configuration
# ===========================================================================
DIAG_DIR = Path("results/workflow_A_npe2d_20260224_104426/diagnostics")
RESULTS_FILE = DIAG_DIR / "diagnostics_results.pkl"
OUTPUT_FILE = DIAG_DIR / "rank_histograms.png"
PARAM_NAMES = ['U', 'P', 'rho']

# ===========================================================================
# Load data
# ===========================================================================
print(f"Loading diagnostics from {RESULTS_FILE}")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

ranks = results['sbc_ranks']  # shape (n_sbc_sims, n_params)
print(f"Ranks shape: {ranks.shape}")

# Convert to numpy
if hasattr(ranks, 'numpy'):
    ranks_np = ranks.numpy()
else:
    ranks_np = np.array(ranks)

n_sbc_sims, n_params = ranks_np.shape
# Infer num_posterior_samples from max rank value
n_posterior_samples = int(ranks_np.max()) + 1
print(f"Number of SBC simulations: {n_sbc_sims}")
print(f"Inferred num_posterior_samples: ~{n_posterior_samples}")
print()

# ===========================================================================
# Rank statistics
# ===========================================================================
# Under perfect calibration, ranks ~ Uniform(0, n_posterior_samples)
# Expected mean = n_posterior_samples / 2
# Expected std  = n_posterior_samples / sqrt(12)
expected_mean = n_posterior_samples / 2.0
expected_std = n_posterior_samples / np.sqrt(12.0)

print("=" * 70)
print("RANK STATISTICS")
print("=" * 70)
print(f"{'Param':>6s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'Skew':>8s}  {'Kurt':>8s}  {'KS_p':>8s}")
print(f"{'expect':>6s}  {expected_mean:8.1f}  {expected_mean:8.1f}  {expected_std:8.1f}  {'0.000':>8s}  {'-1.200':>8s}  {'> 0.05':>8s}")
print("-" * 70)

for j, name in enumerate(PARAM_NAMES):
    r = ranks_np[:, j]
    mean = np.mean(r)
    median = np.median(r)
    std = np.std(r)
    skewness = stats.skew(r)
    kurtosis = stats.kurtosis(r)  # excess kurtosis; uniform => -1.2
    # KS test against uniform
    ks_stat, ks_p = stats.kstest(r / n_posterior_samples, 'uniform')
    print(f"{name:>6s}  {mean:8.1f}  {median:8.1f}  {std:8.1f}  {skewness:8.3f}  {kurtosis:8.3f}  {ks_p:8.4f}")

    # Percentile analysis
    pct_below_25 = np.mean(r < n_posterior_samples * 0.25) * 100
    pct_below_50 = np.mean(r < n_posterior_samples * 0.50) * 100
    pct_below_75 = np.mean(r < n_posterior_samples * 0.75) * 100
    pct_in_tails = (np.mean(r < n_posterior_samples * 0.1) + np.mean(r > n_posterior_samples * 0.9)) * 100
    pct_in_middle = np.mean((r > n_posterior_samples * 0.25) & (r < n_posterior_samples * 0.75)) * 100
    print(f"        Pct below 25%: {pct_below_25:.1f}% (expect 25%)")
    print(f"        Pct below 50%: {pct_below_50:.1f}% (expect 50%)")
    print(f"        Pct below 75%: {pct_below_75:.1f}% (expect 75%)")
    print(f"        Pct in tails (10%/90%): {pct_in_tails:.1f}% (expect 20%)")
    print(f"        Pct in middle (25-75%): {pct_in_middle:.1f}% (expect 50%)")
    print()

# ===========================================================================
# Interpretation
# ===========================================================================
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
for j, name in enumerate(PARAM_NAMES):
    r = ranks_np[:, j]
    skewness = stats.skew(r)
    kurtosis = stats.kurtosis(r)
    mean = np.mean(r)
    mean_deviation = (mean - expected_mean) / expected_std

    issues = []
    if abs(mean_deviation) > 0.5:
        if mean_deviation > 0:
            issues.append(f"BIASED LOW (mean rank {mean:.0f} > expected {expected_mean:.0f}, "
                         f"z={mean_deviation:.2f}) -> posterior systematically below true values")
        else:
            issues.append(f"BIASED HIGH (mean rank {mean:.0f} < expected {expected_mean:.0f}, "
                         f"z={mean_deviation:.2f}) -> posterior systematically above true values")

    if kurtosis < -1.5:
        issues.append(f"OVERCONFIDENT (excess kurtosis {kurtosis:.2f} << -1.2, "
                     f"U-shaped ranks -> posteriors too narrow)")
    elif kurtosis > -0.8:
        issues.append(f"UNDERCONFIDENT (excess kurtosis {kurtosis:.2f} >> -1.2, "
                     f"hump-shaped ranks -> posteriors too wide)")

    if abs(skewness) > 0.3:
        direction = "right-skewed (more high ranks)" if skewness > 0 else "left-skewed (more low ranks)"
        issues.append(f"ASYMMETRIC (skew={skewness:.3f}, {direction})")

    if not issues:
        issues.append("LOOKS OK (no strong deviations)")

    print(f"\n{name}:")
    for issue in issues:
        print(f"  - {issue}")

# ===========================================================================
# Also check: do ranks show structure across SBC sim index?
# (this could indicate ordering/seed effects)
# ===========================================================================
print("\n" + "=" * 70)
print("CORRELATION CHECK: rank vs SBC simulation index")
print("=" * 70)
for j, name in enumerate(PARAM_NAMES):
    r = ranks_np[:, j]
    corr, pval = stats.pearsonr(np.arange(n_sbc_sims), r)
    print(f"  {name}: Pearson r={corr:.4f}, p={pval:.4f}")

# ===========================================================================
# Check cross-parameter rank correlations
# ===========================================================================
print("\n" + "=" * 70)
print("CROSS-PARAMETER RANK CORRELATIONS")
print("=" * 70)
for i in range(n_params):
    for j in range(i+1, n_params):
        corr, pval = stats.pearsonr(ranks_np[:, i], ranks_np[:, j])
        print(f"  {PARAM_NAMES[i]} vs {PARAM_NAMES[j]}: r={corr:.4f}, p={pval:.4f}")

# ===========================================================================
# Plotting
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Rank histograms
for j, name in enumerate(PARAM_NAMES):
    ax = axes[0, j]
    r = ranks_np[:, j]
    n_bins = 30
    ax.hist(r, bins=n_bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    # Expected uniform density
    uniform_density = 1.0 / n_posterior_samples
    ax.axhline(uniform_density, color='red', linestyle='--', linewidth=2, label='Uniform')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}: Rank Histogram')
    ax.legend()

    # Annotate with stats
    mean = np.mean(r)
    skewness = stats.skew(r)
    kurtosis = stats.kurtosis(r)
    ax.text(0.05, 0.95, f'mean={mean:.0f}\nskew={skewness:.2f}\nkurt={kurtosis:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2: Rank ECDFs
for j, name in enumerate(PARAM_NAMES):
    ax = axes[1, j]
    r = ranks_np[:, j]
    # Sort ranks and compute empirical CDF
    sorted_r = np.sort(r) / n_posterior_samples
    ecdf = np.arange(1, n_sbc_sims + 1) / n_sbc_sims

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Uniform')
    ax.plot(sorted_r, ecdf, 'b-', linewidth=2, label='ECDF')
    ax.set_xlabel('Normalized Rank')
    ax.set_ylabel('ECDF')
    ax.set_title(f'{name}: Rank ECDF')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Compute and annotate max deviation
    uniform_cdf = sorted_r  # under uniform, CDF at value v is v
    max_dev_idx = np.argmax(np.abs(ecdf - uniform_cdf))
    max_dev = ecdf[max_dev_idx] - uniform_cdf[max_dev_idx]
    direction = "above" if max_dev > 0 else "below"
    ax.text(0.05, 0.95, f'max dev: {max_dev:+.3f}\n({direction} diagonal)',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Shade the deviation
    ax.fill_between(sorted_r, ecdf, uniform_cdf, alpha=0.2, color='red')

fig.suptitle('Model A 2D: SBC Rank Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nSaved rank histograms to {OUTPUT_FILE}")

# ===========================================================================
# Additional: Load model and check a few posteriors directly
# ===========================================================================
print("\n" + "=" * 70)
print("POSTERIOR SAMPLE CHECK (loading trained model)")
print("=" * 70)

import torch
import sys
sys.path.insert(0, 'src')

try:
    from inference import RandomWalkNPE

    model_path = "results/workflow_A_npe2d_20260224_104426/npe_model.pkl"
    print(f"Loading model from {model_path}")
    npe = RandomWalkNPE.load_model(model_path, device='cpu')

    # Get the SBC thetas and xs
    sbc_thetas = results['sbc_ranks']  # Wait, this is ranks not thetas
    # The diagnostics pickle should also have the posterior_estimator info
    # Let's load the SBC data from the diagnostics run
    # Actually, run_sbc returns ranks and dap_samples, and we stored both
    dap_samples = results.get('sbc_dap_samples', None)
    if dap_samples is not None:
        print(f"DAP samples shape: {dap_samples.shape}")
        dap_np = dap_samples.numpy() if hasattr(dap_samples, 'numpy') else np.array(dap_samples)
        print("\nDAP (Data-Averaged Posterior) sample statistics:")
        print(f"{'Param':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
        for j, name in enumerate(PARAM_NAMES):
            d = dap_np[:, j]
            print(f"{name:>6s}  {np.mean(d):8.4f}  {np.std(d):8.4f}  {np.min(d):8.4f}  {np.max(d):8.4f}")

        # Under perfect calibration, DAP samples should look like the prior
        # For Model A: U ~ U(0.1, 1.0), P ~ U(0.01, 1.0), rho ~ U(0.0, 1.0)
        prior_means = [0.55, 0.505, 0.5]
        prior_stds = [0.9/np.sqrt(12), 0.99/np.sqrt(12), 1.0/np.sqrt(12)]
        print("\nDAP vs Prior comparison:")
        print(f"{'Param':>6s}  {'DAP_mean':>10s}  {'Prior_mean':>10s}  {'DAP_std':>10s}  {'Prior_std':>10s}  {'Mean_diff':>10s}")
        for j, name in enumerate(PARAM_NAMES):
            d = dap_np[:, j]
            dm = np.mean(d)
            ds = np.std(d)
            pm = prior_means[j]
            ps = prior_stds[j]
            print(f"{name:>6s}  {dm:10.4f}  {pm:10.4f}  {ds:10.4f}  {ps:10.4f}  {dm-pm:+10.4f}")

    # Now sample a few posteriors and check their widths
    print("\n--- Sampling posteriors for 5 SBC observations ---")

    # We need the SBC observations. The diagnostics pickle doesn't store them
    # directly (run_sbc takes them as input). Let's regenerate a few or use
    # the leakage data. Actually, let's load the training data and use a few
    # random observations as proxy test cases.
    training_data_path = "results/workflow_A_npe2d_20260205_150346/training_data.pkl"
    print(f"Loading training data from {training_data_path}")
    with open(training_data_path, 'rb') as f:
        train_data = pickle.load(f)
    train_thetas = train_data['parameters']
    train_xs = train_data['observations']
    print(f"Training data: thetas {train_thetas.shape}, xs {train_xs.shape}")

    # Bypass rejection sampling (same as diagnostics.py)
    posterior = npe.posterior
    posterior.posterior_estimator = posterior.posterior_estimator.cpu()

    _flow = posterior.posterior_estimator
    _cond_shape = _flow.condition_shape
    _prior_low = torch.tensor([0.1, 0.01, 0.0], dtype=torch.float32)
    _prior_high = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    def _ensure_batch_dim(x):
        if x.dim() == len(_cond_shape):
            return x.unsqueeze(0)
        return x

    @torch.no_grad()
    def _direct_sample(sample_shape=torch.Size(), x=None, **kw):
        x = posterior._x_else_default_x(x)
        x = _ensure_batch_dim(x)
        n = torch.Size(sample_shape).numel()
        raw = _flow.sample((n,), condition=x)[:, 0]
        clamped = torch.clamp(raw, min=_prior_low, max=_prior_high)
        return clamped

    posterior.sample = _direct_sample

    # Check a few training examples with known true params
    np.random.seed(42)
    test_indices = np.random.choice(len(train_thetas), 10, replace=False)

    print(f"\n{'Idx':>5s}  {'Param':>5s}  {'True':>8s}  {'Post_mean':>10s}  {'Post_std':>10s}  {'Post_5%':>8s}  {'Post_95%':>8s}  {'Contains':>8s}")
    print("-" * 85)

    posterior_widths = {name: [] for name in PARAM_NAMES}
    posterior_biases = {name: [] for name in PARAM_NAMES}

    for idx in test_indices:
        x_obs = train_xs[idx].unsqueeze(0) if train_xs[idx].dim() == 2 else train_xs[idx]
        if x_obs.dim() == 2:
            x_obs = x_obs.unsqueeze(0)  # add batch dim for 2D
        true_theta = train_thetas[idx].numpy()

        samples = posterior.sample((2000,), x=x_obs)
        samples_np = samples.numpy()

        for j, name in enumerate(PARAM_NAMES):
            s = samples_np[:, j]
            mean = np.mean(s)
            std = np.std(s)
            q5 = np.percentile(s, 5)
            q95 = np.percentile(s, 95)
            contains = "YES" if q5 <= true_theta[j] <= q95 else "NO"
            print(f"{idx:5d}  {name:>5s}  {true_theta[j]:8.4f}  {mean:10.4f}  {std:10.4f}  {q5:8.4f}  {q95:8.4f}  {contains:>8s}")
            posterior_widths[name].append(q95 - q5)
            posterior_biases[name].append(mean - true_theta[j])
        print()

    print("\nSummary of posterior widths (90% CI width):")
    for name in PARAM_NAMES:
        widths = posterior_widths[name]
        prior_range = {'U': 0.9, 'P': 0.99, 'rho': 1.0}[name]
        mean_width = np.mean(widths)
        print(f"  {name}: mean 90% CI width = {mean_width:.4f} (prior range = {prior_range:.2f}, "
              f"fraction = {mean_width/prior_range:.2f})")

    print("\nSummary of posterior bias (mean - true):")
    for name in PARAM_NAMES:
        biases = posterior_biases[name]
        mean_bias = np.mean(biases)
        std_bias = np.std(biases)
        print(f"  {name}: mean bias = {mean_bias:+.4f} +/- {std_bias:.4f}")

except Exception as e:
    import traceback
    print(f"Error during posterior check: {e}")
    traceback.print_exc()

print("\nDone.")
