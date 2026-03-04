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

    def is_reparameterized(self) -> bool:
        """True if this config is a reparameterized (continuum-space) variant."""
        return self.name.endswith('_reparam')

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
        param_names=['U', 'P', 'R'],
        param_labels=['U (initial occupancy)', 'P (movement prob.)', 'R (proliferation prob.)'],
        prior_low=[0.1, 0.01, 0.001],
        prior_high=[1.0, 1.0, 0.2],
        has_exclusion=True,
        has_bias=False,
        has_growth=True,
        continuum_names=['D', 'r'],
        continuum_labels=[r'$D$ (diffusivity)', r'$r$ (proliferation rate)'],
    ),

    'C': ModelConfig(
        name='C',
        description='Exclusion + bias + growth',
        param_names=['U', 'P', 'rho', 'R'],
        param_labels=['U (initial occupancy)', 'P (movement prob.)', r'$\rho$ (bias)', 'R (proliferation prob.)'],
        prior_low=[0.1, 0.01, 0.0, 0.001],
        prior_high=[1.0, 1.0, 1.0, 0.2],
        has_exclusion=True,
        has_bias=True,
        has_growth=True,
        continuum_names=['D', 'v', 'r'],
        continuum_labels=[r'$D$ (diffusivity)', r'$v$ (drift velocity)', r'$r$ (proliferation rate)'],
    ),

    'A_reparam': ModelConfig(
        name='A_reparam',
        description='Model A reparameterized to continuum (U, D, v)',
        param_names=['U', 'D', 'v'],
        param_labels=[r'$U$', r'$D$ (diffusivity)', r'$v$ (drift velocity)'],
        prior_low=[0.1, 0.0025, 0.0],
        prior_high=[1.0, 0.25, 0.5],
        has_exclusion=True,
        has_bias=True,
        has_growth=False,
        continuum_names=[],
        continuum_labels=[],
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


# ---------------------------------------------------------------------------
# Lattice <-> Continuum transforms (Delta=tau=1)
# ---------------------------------------------------------------------------

def lattice_to_continuum_theta(theta: np.ndarray, model_name: str = 'A') -> np.ndarray:
    """
    Transform lattice parameters to continuum parameters.

    For Model A: (U, P, rho) -> (U, D, v) where D = P/4, v = P*rho/2.

    Parameters
    ----------
    theta : np.ndarray, shape (..., n_params)
        Lattice-space parameters.
    model_name : str
        Base model name (currently only 'A' supported).

    Returns
    -------
    np.ndarray, same shape as theta
        Continuum-space parameters.
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(theta)

    if model_name == 'A':
        U = theta[..., 0]
        P = theta[..., 1]
        rho = theta[..., 2]
        out[..., 0] = U
        out[..., 1] = P / 4.0          # D = P*Delta^2 / (4*tau)
        out[..., 2] = P * rho / 2.0    # v = P*rho*Delta / (2*tau)
    else:
        raise ValueError(f"lattice_to_continuum_theta not implemented for model '{model_name}'")

    return out.astype(np.float32)


def continuum_to_lattice_theta(theta: np.ndarray, model_name: str = 'A') -> np.ndarray:
    """
    Transform continuum parameters back to lattice parameters.

    For Model A: (U, D, v) -> (U, P, rho) where P = 4D, rho = v/(2D).
    rho is clamped to [0, 1].

    Parameters
    ----------
    theta : np.ndarray, shape (..., n_params)
        Continuum-space parameters.
    model_name : str
        Base model name (currently only 'A' supported).

    Returns
    -------
    np.ndarray, same shape as theta
        Lattice-space parameters.
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(theta)

    if model_name == 'A':
        U = theta[..., 0]
        D = theta[..., 1]
        v = theta[..., 2]
        P = 4.0 * D
        # Avoid division by zero: where D is tiny, set rho=0
        rho = np.where(D > 1e-10, v / (2.0 * D), 0.0)
        rho = np.clip(rho, 0.0, 1.0)
        P = np.clip(P, 0.0, 1.0)
        out[..., 0] = U
        out[..., 1] = P
        out[..., 2] = rho
    else:
        raise ValueError(f"continuum_to_lattice_theta not implemented for model '{model_name}'")

    return out.astype(np.float32)
