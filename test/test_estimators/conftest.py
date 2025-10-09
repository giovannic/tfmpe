"""Shared fixtures for estimator tests."""

from jaxtyping import Array
import diffrax
import jax.numpy as jnp
import pytest

from tfmpe.preprocessing.tokens import Tokens

@pytest.fixture
def doubling_vf():
    """Vector field for doubling flow: f(context, params, t)
    -> Array.

    For testing, ignores context and applies log(2) · params.data.
    """
    def vf(tokens: Tokens, t):
        return jnp.log(2.0) * tokens.data[:, tokens.partition_idx:]
    return vf


@pytest.fixture
def solver():
    """Diffrax ODE solver instance."""
    return diffrax.Dopri5()


def create_mock_tokens(
        context: Array,
        params: Array,
        sample_ndims:int = 0
    ):
    return Tokens.from_pytree(
        {'x': params, 'y': context},
        condition=['y'],
        sample_ndims=sample_ndims
    )
