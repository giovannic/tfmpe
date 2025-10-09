"""Shared fixtures for transformer tests."""

import jax.numpy as jnp
import pytest

from tfmpe.preprocessing import Tokens, Labeller


@pytest.fixture
def simple_labeller() -> Labeller:
    """Labeller for simple hierarchical structure."""
    return Labeller(label_map={'mu': 0, 'theta': 1, 'obs': 2})


@pytest.fixture
def simple_tokens(simple_labeller: Labeller) -> Tokens:
    """Create context tokens with mu and obs."""
    pytree = {
        'mu': jnp.array([[[1.0]]]),
        'obs': jnp.array([[[0.1], [0.2]]]),
        'theta': jnp.array([[[2.0], [3.0]]]),
    }
    return Tokens.from_pytree(
        pytree,
        labeller=simple_labeller,
        condition=['obs'],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'obs': 1},
        pad_to_even=False,
    )
