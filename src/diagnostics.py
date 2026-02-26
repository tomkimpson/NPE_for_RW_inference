"""
SBI diagnostics module: Simulation-Based Calibration (SBC) and expected coverage (TARP).

Provides functions to generate calibration data and run standard SBI
diagnostics for trained NPE posteriors.
"""

import numpy as np
import torch
import pickle
import time
import os
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple

from sbi.utils import BoxUniform
from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
from sbi.analysis import sbc_rank_plot

from models import ModelConfig, get_model_config, lattice_to_continuum_theta
from simulator import RandomWalkSimulator, ExclusionRandomWalkSimulator
from inference import _init_worker, _run_single_sim, RandomWalkNPE
from utils import configure_warnings

configure_warnings()


def generate_sbc_data(
    model_config: ModelConfig,
    Lx: int,
    Ly: int,
    T: int,
    initial_region_half_width: int,
    n_sims: int,
    n_workers: int = 8,
    use_2d: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample parameter vectors from the prior and simulate observations for SBC.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration (prior bounds, flags, fixed params).
    Lx, Ly, T, initial_region_half_width : int
        Simulation grid and time parameters.
    n_sims : int
        Number of simulations to generate.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    thetas : torch.Tensor of shape (n_sims, n_params)
    xs : torch.Tensor of shape (n_sims, Lx) or (n_sims, Ly*Lx) if use_2d
    """
    is_reparam = model_config.is_reparameterized()

    prior = BoxUniform(
        low=torch.tensor(model_config.prior_low, dtype=torch.float32),
        high=torch.tensor(model_config.prior_high, dtype=torch.float32),
    )

    if is_reparam and model_config.name == 'A_reparam':
        # Sample (U, D, v) from box prior, reject where rho = v/(2D) > 1
        from models import continuum_to_lattice_theta
        accepted = []
        batch_size = n_sims * 2  # oversample to account for rejections
        while len(accepted) < n_sims:
            candidates = prior.sample((batch_size,)).numpy()
            lattice = continuum_to_lattice_theta(candidates, model_name='A')
            # Valid samples: rho <= 1 (before clamping)
            D = candidates[:, 1].astype(np.float64)
            v = candidates[:, 2].astype(np.float64)
            rho_raw = np.where(D > 1e-10, v / (2.0 * D), 0.0)
            valid = rho_raw <= 1.0
            accepted.extend(candidates[valid][:n_sims - len(accepted)])
        thetas_np = np.array(accepted[:n_sims], dtype=np.float32)
        thetas = torch.tensor(thetas_np)
        # Convert to lattice for simulation
        thetas_lattice = continuum_to_lattice_theta(thetas_np, model_name='A')
        lattice_cfg = get_model_config('A')
        param_names_sim = lattice_cfg.param_names
        fixed_params_sim = lattice_cfg.fixed_params
        print(f"[REPARAM] SBC: sampled {n_sims} valid (U,D,v) points "
              f"(rejection rate: rho>1 excluded from box prior)")
    else:
        thetas = prior.sample((n_sims,))
        thetas_np = thetas.numpy()
        thetas_lattice = None
        param_names_sim = model_config.param_names
        fixed_params_sim = model_config.fixed_params

    # Build simulator
    use_exclusion = model_config.has_exclusion
    if use_exclusion:
        sim_class_name = 'ExclusionRandomWalkSimulator'
        sim_kwargs = {
            'Lx': Lx, 'Ly': Ly,
            'initial_region_half_width': initial_region_half_width,
            'has_bias': model_config.has_bias,
            'has_growth': model_config.has_growth,
        }
    else:
        sim_class_name = 'RandomWalkSimulator'
        sim_kwargs = {
            'Lx': Lx, 'Ly': Ly,
            'initial_region_half_width': initial_region_half_width,
        }

    sim_thetas = thetas_lattice if thetas_lattice is not None else thetas_np

    sim_args_list = []
    for i in range(n_sims):
        param_values = sim_thetas[i].tolist()
        seed_i = 90000 + i  # deterministic seeds for reproducibility
        sim_args_list.append((
            i, param_values, param_names_sim,
            fixed_params_sim, use_exclusion, T, seed_i, use_2d,
        ))

    observations = np.zeros((n_sims, Ly, Lx)) if use_2d else np.zeros((n_sims, Lx))

    try:
        parent_cpus = set(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        parent_cpus = None

    ctx = mp.get_context('fork')
    t0 = time.time()
    print(f"Generating {n_sims} SBC simulations using {n_workers} workers...")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(sim_class_name, sim_kwargs, parent_cpus),
    ) as executor:
        futures = {executor.submit(_run_single_sim, args): args[0] for args in sim_args_list}
        done_count = 0
        log_interval = max(1, n_sims // 10)
        for future in as_completed(futures):
            idx, obs = future.result()
            observations[idx] = obs
            done_count += 1
            if done_count % log_interval == 0 or done_count == n_sims:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"   [{done_count}/{n_sims}] {elapsed:.1f}s ({rate:.1f} sims/s)")

    xs = torch.tensor(observations, dtype=torch.float32)
    return thetas, xs


def run_all_diagnostics(
    posterior,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    param_names: list,
    n_posterior_samples: int = 1000,
    output_dir: str = "diagnostics",
) -> Dict[str, Any]:
    """
    Run SBC and TARP diagnostics, save plots and summary.

    Parameters
    ----------
    posterior : sbi posterior object
        Trained posterior (must support .sample and .log_prob).
    thetas : torch.Tensor (n_sims, n_params)
        Prior samples used for SBC.
    xs : torch.Tensor (n_sims, obs_dim)
        Simulated observations corresponding to thetas.
    param_names : list of str
        Parameter names for labelling.
    n_posterior_samples : int
        Number of posterior samples per SBC simulation.
    output_dir : str
        Directory to save outputs.

    Returns
    -------
    results : dict
        Dictionary with all diagnostic outputs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Run diagnostics on CPU to avoid CUDA-only op limitations
    # (e.g. torch.histogram is CPU-only). Move the posterior estimator
    # and prior to CPU; thetas/xs are already CPU tensors.
    posterior.posterior_estimator = posterior.posterior_estimator.cpu()
    if hasattr(posterior, 'prior') and posterior.prior is not None:
        from sbi.utils import BoxUniform
        low = posterior.prior.base_dist.low.cpu()
        high = posterior.prior.base_dist.high.cpu()
        posterior.prior = BoxUniform(low=low, high=high)
    thetas = thetas.cpu()
    xs = xs.cpu()

    # Bypass sbi's rejection sampling in posterior.sample() /
    # sample_batched().  DirectPosterior rejects any flow sample
    # outside the prior box, which can stall if the flow has even
    # modest leakage.  Instead, we draw raw flow samples and clamp
    # them to the prior bounds.  This preserves the speed advantage
    # of direct sampling while ensuring samples respect the prior
    # support as required by SBC theory (Talts et al. 2018).
    _flow = posterior.posterior_estimator
    _cond_shape = _flow.condition_shape
    _prior_low = low   # already on CPU from prior reconstruction above
    _prior_high = high

    def _ensure_batch_dim(x):
        """Add batch dimension if missing."""
        if x.dim() == len(_cond_shape):
            return x.unsqueeze(0)
        return x

    @torch.no_grad()
    def _direct_sample(sample_shape=torch.Size(), x=None, **kw):
        x = posterior._x_else_default_x(x)
        x = _ensure_batch_dim(x)
        n = torch.Size(sample_shape).numel()
        raw = _flow.sample((n,), condition=x)[:, 0]
        return torch.clamp(raw, min=_prior_low, max=_prior_high)

    @torch.no_grad()
    def _direct_sample_batched(sample_shape, x, **kw):
        x = _ensure_batch_dim(x)
        n = torch.Size(sample_shape).numel()
        raw = _flow.sample((n,), condition=x)
        return torch.clamp(raw, min=_prior_low, max=_prior_high)

    posterior.sample = _direct_sample
    posterior.sample_batched = _direct_sample_batched
    print("Running diagnostics on CPU (direct flow sampling, clamped to prior bounds)")

    results = {}

    # ------------------------------------------------------------------
    # SBC
    # ------------------------------------------------------------------
    print("\n--- Running SBC ---")
    ranks, dap_samples = run_sbc(
        thetas, xs, posterior,
        num_posterior_samples=n_posterior_samples,
        reduce_fns="marginals",
        show_progress_bar=True,
        use_batched_sampling=False,
    )
    results['sbc_ranks'] = ranks
    results['sbc_dap_samples'] = dap_samples

    print("Checking SBC results...")
    sbc_checks = check_sbc(ranks, thetas, dap_samples, n_posterior_samples)
    results['sbc_checks'] = sbc_checks
    print(f"  KS p-values: {sbc_checks['ks_pvals']}")
    print(f"  C2ST ranks: {sbc_checks['c2st_ranks']}")
    print(f"  C2ST DAP:   {sbc_checks['c2st_dap']}")

    # ------------------------------------------------------------------
    # Flow leakage measurement
    # ------------------------------------------------------------------
    print("\n--- Measuring flow leakage outside prior bounds ---")
    n_leakage_obs = min(100, xs.shape[0])
    n_leakage_samples = 1000
    leakage_below = torch.zeros(len(param_names))
    leakage_above = torch.zeros(len(param_names))
    with torch.no_grad():
        for i in range(n_leakage_obs):
            x_i = xs[i].unsqueeze(0)
            raw_samples = _flow.sample((n_leakage_samples,), condition=x_i)[:, 0]
            for j in range(len(param_names)):
                leakage_below[j] += (raw_samples[:, j] < _prior_low[j]).sum().float()
                leakage_above[j] += (raw_samples[:, j] > _prior_high[j]).sum().float()
    total_samples = n_leakage_obs * n_leakage_samples
    leakage_below_frac = (leakage_below / total_samples).numpy()
    leakage_above_frac = (leakage_above / total_samples).numpy()
    leakage_total_frac = leakage_below_frac + leakage_above_frac
    results['leakage_below_frac'] = leakage_below_frac
    results['leakage_above_frac'] = leakage_above_frac
    results['leakage_total_frac'] = leakage_total_frac
    for j, name in enumerate(param_names):
        print(f"  {name}: {leakage_total_frac[j]*100:.2f}% total "
              f"({leakage_below_frac[j]*100:.2f}% below, "
              f"{leakage_above_frac[j]*100:.2f}% above)")

    print("\nPlotting SBC ranks...")
    fig_sbc, _ = sbc_rank_plot(
        ranks, n_posterior_samples,
        parameter_labels=param_names,
    )
    fig_sbc.savefig(out / "sbc_rank_plot.png", dpi=150, bbox_inches='tight')
    plt.close(fig_sbc)
    print(f"  Saved {out / 'sbc_rank_plot.png'}")

    # ------------------------------------------------------------------
    # TARP
    # ------------------------------------------------------------------
    print("\n--- Running TARP ---")
    ecp, alpha = run_tarp(
        thetas, xs, posterior,
        num_posterior_samples=n_posterior_samples,
        show_progress_bar=True,
        use_batched_sampling=False,
    )
    results['tarp_ecp'] = ecp
    results['tarp_alpha'] = alpha

    print("Checking TARP results...")
    atc, ks_pval = check_tarp(ecp, alpha)
    results['tarp_atc'] = atc
    results['tarp_ks_pval'] = ks_pval
    print(f"  ATC: {atc:.4f}")
    print(f"  KS p-value: {ks_pval:.4f}")

    # Plot TARP: ECP vs alpha with diagonal reference
    print("Plotting TARP coverage...")
    fig_tarp, ax_tarp = plt.subplots(1, 1, figsize=(5, 5))
    alpha_np = alpha.cpu().numpy()
    ecp_np = ecp.cpu().numpy()
    ax_tarp.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')
    ax_tarp.plot(alpha_np, ecp_np, 'b-', lw=2, label='TARP')
    ax_tarp.set_xlabel('Credibility level (alpha)')
    ax_tarp.set_ylabel('Expected coverage probability')
    ax_tarp.set_title(f'TARP coverage (ATC={atc:.3f}, KS p={ks_pval:.3f})')
    ax_tarp.legend(loc='lower right')
    ax_tarp.set_xlim(0, 1)
    ax_tarp.set_ylim(0, 1)
    ax_tarp.set_aspect('equal')
    fig_tarp.savefig(out / "tarp_plot.png", dpi=150, bbox_inches='tight')
    plt.close(fig_tarp)
    print(f"  Saved {out / 'tarp_plot.png'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ks_pvals = sbc_checks['ks_pvals'].cpu().numpy()
    c2st_ranks = sbc_checks['c2st_ranks'].cpu().numpy()
    c2st_dap = sbc_checks['c2st_dap'].cpu().numpy()

    summary_lines = [
        "SBI Diagnostics Summary",
        "=" * 40,
        "",
        "SBC (Simulation-Based Calibration)",
        "-" * 40,
    ]
    for i, name in enumerate(param_names):
        ks_pass = "PASS" if ks_pvals[i] > 0.05 else "FAIL"
        c2st_r_pass = "PASS" if abs(c2st_ranks[i] - 0.5) < 0.1 else "FAIL"
        c2st_d_pass = "PASS" if abs(c2st_dap[i] - 0.5) < 0.1 else "FAIL"
        summary_lines.append(
            f"  {name}: KS p={ks_pvals[i]:.4f} [{ks_pass}]  "
            f"C2ST_ranks={c2st_ranks[i]:.4f} [{c2st_r_pass}]  "
            f"C2ST_dap={c2st_dap[i]:.4f} [{c2st_d_pass}]"
        )

    summary_lines += [
        "",
        "TARP (Expected Coverage)",
        "-" * 40,
        f"  ATC: {atc:.4f} {'PASS' if abs(atc) < 0.1 else 'FAIL'}",
        f"  KS p-value: {ks_pval:.4f} {'PASS' if ks_pval > 0.05 else 'FAIL'}",
        "",
        "Flow Leakage (raw samples outside prior bounds)",
        "-" * 40,
    ]
    for i, name in enumerate(param_names):
        summary_lines.append(
            f"  {name}: {leakage_total_frac[i]*100:.2f}% total "
            f"(below: {leakage_below_frac[i]*100:.2f}%, above: {leakage_above_frac[i]*100:.2f}%)"
        )
    summary_lines += [
        "",
        "Criteria: KS p > 0.05, C2ST_ranks ~ 0.5 (+/- 0.1), C2ST_dap ~ 0.5 (+/- 0.1), ATC ~ 0 (+/- 0.1)",
        "Note: Flow samples are clamped to prior bounds before SBC rank computation.",
    ]

    summary_text = "\n".join(summary_lines)
    (out / "diagnostics_summary.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\nSummary saved to {out / 'diagnostics_summary.txt'}")

    # Save raw results
    with open(out / "diagnostics_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"Raw results saved to {out / 'diagnostics_results.pkl'}")

    return results
