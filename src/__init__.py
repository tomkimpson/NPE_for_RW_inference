"""
NPE for Random Walk Inference

A package for Neural Posterior Estimation applied to random walk models in biology.
"""

__version__ = "0.1.0"

from . import simulator
from . import inference
from . import utils

__all__ = ["simulator", "inference", "utils"]