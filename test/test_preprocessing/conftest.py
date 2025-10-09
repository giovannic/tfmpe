"""Shared fixtures for preprocessing tests."""

import jax.numpy as jnp
import pytest

@pytest.fixture
def simple_pytree():
    """Simple pytree with 3 keys, no sample dimensions.

    Structure: mu (1 event, 1 batch), theta (3 events, 1 batch),
    obs (3 events, 1 batch). Total tokens: 7.
    """
    return {
        'mu': jnp.array([[1.0]]),  # (1, 1)
        'theta': jnp.array([[2.0], [3.0], [4.0]]),  # (3, 1)
        'obs': jnp.array([[0.1], [0.2], [0.3]])  # (3, 1)
    }
