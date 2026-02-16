"""
ABC inference module using Sequential Monte Carlo ABC (SMCABC).

Provides an ABC baseline for comparison with NPE, using sbi's SMCABC
implementation.
"""

import numpy as np
import torch
import pickle
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from sbi.inference import SMCABC
from sbi.utils import BoxUniform

from simulator import RandomWalkSimulator, ExclusionRandomWalkSimulator
from models import ModelConfig


def make_sbi_simulator(simulator, param_names, fixed_params, T, use_exclusion):
    """
    Wrap our simulator into sbi's expected interface.

    sbi expects: callable that takes theta tensor (batch, n_params) and
    returns x tensor (batch, x_dim).  Each call must be stochastic
    (no fixed random_seed) so ABC gets independent draws.

    Parameters
    ----------
    simulator : RandomWalkSimulator or ExclusionRandomWalkSimulator
    param_names : list of str
    fixed_params : dict
    T : int
    use_exclusion : bool

    Returns
    -------
    callable
        Function with signature (theta: Tensor) -> Tensor.
    """

    def sbi_simulator(theta):
        # theta may be 1-D (single sample) or 2-D (batch)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)

        batch_size = theta.shape[0]
        results = []

        for i in range(batch_size):
            param_values = theta[i].cpu().numpy()
            theta_dict = dict(zip(param_names, param_values.tolist()))
            theta_dict.update(fixed_params)

            if use_exclusion:
                column_counts, _, _ = simulator.simulate(
                    theta_dict, T, random_seed=None
                )
            else:
                column_counts, _, _ = simulator.simulate(
                    U=theta_dict['U'], P=theta_dict['P'],
                    T=T, random_seed=None
                )
            results.append(torch.tensor(column_counts, dtype=torch.float32))

        return torch.stack(results)

    return sbi_simulator


class RandomWalkABC:
    """SMCABC inference for random walk models."""

    def __init__(self, model_config=None, device='cpu', seed=None):
        """
        Parameters
        ----------
        model_config : ModelConfig, optional
            Model configuration. If None, uses original [U, P] prior.
        device : str
            Device (ABC is CPU-only but we keep the interface consistent).
        seed : int, optional
            Random seed.
        """
        self.model_config = model_config
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if model_config is not None:
            self.param_names = model_config.param_names
            self.prior = BoxUniform(
                low=torch.tensor(model_config.prior_low, dtype=torch.float32),
                high=torch.tensor(model_config.prior_high, dtype=torch.float32),
            )
        else:
            self.param_names = ['U', 'P']
            self.prior = BoxUniform(
                low=torch.tensor([0.01, 0.0]),
                high=torch.tensor([1.0, 1.0]),
            )

    def run(
        self,
        simulator,
        x_obs,
        T,
        num_particles=500,
        num_simulations=50000,
        num_initial_pop=2000,
        epsilon_decay=0.5,
        num_workers=1,
    ):
        """
        Run SMCABC and return posterior samples.

        Parameters
        ----------
        simulator : RandomWalkSimulator or ExclusionRandomWalkSimulator
        x_obs : torch.Tensor
            Observed data of shape (1, Lx).
        T : int
            Simulation time steps.
        num_particles : int
            Particles per population.
        num_simulations : int
            Total simulation budget.
        num_initial_pop : int
            Simulations for the initial population.
        epsilon_decay : float
            Acceptance-threshold decay factor.
        num_workers : int
            Parallel workers for simulation.

        Returns
        -------
        dict with keys:
            'samples' : np.ndarray (num_particles, n_params)
            'elapsed'  : float (wall-clock seconds)
            'summary'  : dict (SMCABC summary)
        """
        cfg = self.model_config
        use_exclusion = isinstance(simulator, ExclusionRandomWalkSimulator)
        fixed_params = cfg.fixed_params if cfg is not None else {}

        sbi_sim = make_sbi_simulator(
            simulator, self.param_names, fixed_params, T, use_exclusion
        )

        smcabc = SMCABC(
            simulator=sbi_sim,
            prior=self.prior,
            distance='l2',
            num_workers=num_workers,
            simulation_batch_size=1,
            show_progress_bars=True,
        )

        print(f"Running SMCABC: {num_simulations} simulations, "
              f"{num_particles} particles, epsilon_decay={epsilon_decay}")
        start = time.time()

        samples = smcabc(
            x_o=x_obs,
            num_particles=num_particles,
            num_initial_pop=num_initial_pop,
            num_simulations=num_simulations,
            epsilon_decay=epsilon_decay,
            return_summary=False,
        )

        elapsed = time.time() - start

        # samples is a tensor of shape (num_particles, n_params)
        samples_np = samples.cpu().numpy()

        print(f"SMCABC completed in {elapsed:.1f}s")
        print(f"   Accepted {samples_np.shape[0]} particles")
        for pidx, pname in enumerate(self.param_names):
            pmean = samples_np[:, pidx].mean()
            pstd = samples_np[:, pidx].std()
            print(f"   {pname}: {pmean:.4f} +/- {pstd:.4f}")

        return {
            'samples': samples_np,
            'elapsed': elapsed,
        }

    def save_results(self, results, filepath, metadata=None):
        """Save ABC results to pickle."""
        data = {
            'posterior_samples': results['samples'],
            'param_names': self.param_names,
            'method': 'SMCABC',
            'elapsed': results['elapsed'],
            'metadata': metadata or {},
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved ABC results to {filepath}")
