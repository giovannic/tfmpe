"""Training loops for TFMPE models.

Provides speed-optimized and memory-efficient training implementations.
"""

import jax
from jax import numpy as jnp
from jaxtyping import Array

from ..tfmpe import TFMPE
from ...preprocessing.tokens import Tokens

import dataclasses

from .bottom_up import fit_bottom_up
from .directly import fit_directly
from .posterior_factorisation import fit_pf
from .loss import cfm_loss

__all__ = [
    "cfm_loss",
    "fit_bottom_up",
    "fit_directly",
    "fit_pf",
]
