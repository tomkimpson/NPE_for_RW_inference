"""
Model configuration registry for random walk models.

Defines parameter spaces, prior bounds, and continuum relationships
for each model variant following Simpson & Planck (2025) conventions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a random walk model variant."""

    name: str
    description: str

    # Parameter names (lattice-space, used by NPE)
    param_names: List[str]
    param_labels: List[str]

    # Prior bounds (one per parameter, matching param_names order)
    prior_low: List[float]
    prior_high: List[float]

    # Fixed parameters not inferred by NPE (e.g. U fixed for growth models)
    fixed_params: Dict[str, float] = field(default_factory=dict)

    # Model flags
    has_exclusion: bool = False
    has_bias: bool = False
    has_growth: bool = False

    # Continuum-space parameter names (for reporting)
    continuum_names: List[str] = field(default_factory=list)
    continuum_labels: List[str] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def to_continuum(self, theta_lattice: np.ndarray, Delta: float = 1.0, tau: float = 1.0) -> np.ndarray:
        """
        Convert lattice parameters to continuum parameters for reporting.

        Parameters
        ----------
        theta_lattice : np.ndarray
            Lattice-space parameters, shape (..., n_params).
        Delta : float
            Lattice spacing.
        tau : float
            Timestep duration.

        Returns
        -------
        np.ndarray
            Continuum-space parameters in the order of self.continuum_names.
        """
        lookup = dict(zip(self.param_names, range(self.n_params)))
        results = []

        for cname in self.continuum_names:
            if cname == 'D':
                P = theta_lattice[..., lookup['P']]
                results.append(P * Delta**2 / (4 * tau))
            elif cname == 'v':
                P = theta_lattice[..., lookup['P']]
                rho = theta_lattice[..., lookup['rho']]
                results.append(P * rho * Delta / (2 * tau))
            elif cname == 'r':
                R = theta_lattice[..., lookup['R']]
                results.append(R / tau)
            else:
                raise ValueError(f"Unknown continuum parameter: {cname}")

        return np.stack(results, axis=-1)

    def get_all_params(self, theta_inferred: Dict[str, float]) -> Dict[str, float]:
        """
        Merge inferred parameters with fixed parameters into a complete dict.

        Parameters
        ----------
        theta_inferred : dict
            Inferred parameter values keyed by name.

        Returns
        -------
        dict
            Complete parameter dict including fixed values.
        """
        params = dict(self.fixed_params)
        params.update(theta_inferred)
        return params


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, ModelConfig] = {

    'original': ModelConfig(
        name='original',
        description='Unbiased diffusion without exclusion or growth',
        param_names=['U', 'P'],
        param_labels=['U (initial occupancy)', 'P (movement prob.)'],
        prior_low=[0.01, 0.0],
        prior_high=[1.0, 1.0],
        has_exclusion=False,
        has_bias=False,
        has_growth=False,
        continuum_names=['D'],
        continuum_labels=[r'$D$ (diffusivity)'],
    ),

    'A': ModelConfig(
        name='A',
        description='Exclusion + bias, no growth',
        param_names=['U', 'P', 'rho'],
        param_labels=['U (initial occupancy)', 'P (movement prob.)', r'$\rho$ (bias)'],
        prior_low=[0.1, 0.01, 0.0],
        prior_high=[1.0, 1.0, 1.0],
        has_exclusion=True,
        has_bias=True,
        has_growth=False,
        continuum_names=['D', 'v'],
        continuum_labels=[r'$D$ (diffusivity)', r'$v$ (drift velocity)'],
    ),

    'B': ModelConfig(
        name='B',
        description='Exclusion + growth (Fisher-Kolmogorov), no bias',
        param_names=['P', 'R'],
        param_labels=['P (movement prob.)', 'R (proliferation prob.)'],
        prior_low=[0.01, 0.001],
        prior_high=[1.0, 0.2],
        fixed_params={'U': 0.5},
        has_exclusion=True,
        has_bias=False,
        has_growth=True,
        continuum_names=['D', 'r'],
        continuum_labels=[r'$D$ (diffusivity)', r'$r$ (proliferation rate)'],
    ),

    'C': ModelConfig(
        name='C',
        description='Exclusion + bias + growth',
        param_names=['P', 'rho', 'R'],
        param_labels=['P (movement prob.)', r'$\rho$ (bias)', 'R (proliferation prob.)'],
        prior_low=[0.01, 0.0, 0.001],
        prior_high=[1.0, 1.0, 0.2],
        fixed_params={'U': 0.5},
        has_exclusion=True,
        has_bias=True,
        has_growth=True,
        continuum_names=['D', 'v', 'r'],
        continuum_labels=[r'$D$ (diffusivity)', r'$v$ (drift velocity)', r'$r$ (proliferation rate)'],
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """
    Retrieve a model configuration by name.

    Parameters
    ----------
    name : str
        One of 'original', 'A', 'B', 'C'.

    Returns
    -------
    ModelConfig

    Raises
    ------
    ValueError
        If model name is not recognised.
    """
    if name not in MODEL_CONFIGS:
        valid = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{name}'. Valid models: {valid}")
    return MODEL_CONFIGS[name]
