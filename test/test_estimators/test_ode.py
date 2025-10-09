"""Unit tests for ODE solving helpers."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import pytest

from tfmpe.estimators.ode import (
    solve_forward_ode,
    solve_augmented_ode,
)

from .conftest import create_mock_tokens

class TestSolveForwardODE:
    """Test forward ODE solver with doubling flow."""

    @pytest.mark.parametrize(
        "n_tokens,batch_size",
        [
            (1, 1),
            (2, 1),
            (1, 3),
            (2, 2),
        ],
    )
    def test_multidim_tokens_shapes(
        self, doubling_vf, solver, n_tokens, batch_size
    ):
        """Test forward ODE with different Tokens shapes.

        Verify that:
        1. Output shape matches input shape
        2. Output is doubled (doubling flow: f(x) = log(2)·x)
        """
        # Create data with specified shape (n_tokens, batch)
        data_shape = (n_tokens, batch_size)
        seed = jax.random.PRNGKey(42)
        x0_data = jax.random.normal(seed, data_shape)

        # Create mock context and params
        tokens = create_mock_tokens(
            jnp.zeros((n_tokens, batch_size)),
            x0_data
        )

        # Solve forward ODE
        result = solve_forward_ode(
            doubling_vf,
            tokens,
            solver,
        )

        # Check output shape matches input shape
        assert result.data.shape == tokens.data.shape

        # Check that values are doubled (analytical solution for
        # doubling flow is 2x)
        assert jnp.allclose(
            result.data[result.partition_idx:], 2.0 * x0_data, rtol=0.01
        )

class TestSolveAugmentedODE:
    """Test augmented ODE with trace-based log determinant."""

    def test_constant_vf_preserves_density(self, solver):
        """Test that constant VF (f=0) preserves Gaussian density.

        For constant transform f(x)=0, the flow is identity.
        Sample density should equal base density N(0,1).
        """
        def constant_vf(tok, t):
            return jnp.zeros_like(tok.data[:, tok.partition_idx:])

        # Create 100 sample Tokens from N(0, 1), n_tokens=1, batch=1
        seed = jax.random.PRNGKey(42)
        n_samples = 100
        samples = jax.random.normal(seed, (n_samples, 1, 1))

        # Compute expected log prob for N(0, 1)
        expected_log_probs = norm.logpdf(samples[:, 0, 0])

        tokens = create_mock_tokens(
            jnp.zeros((n_samples, 1, 1)),
            samples,
            sample_ndims=1
        )
        rngs = jnp.stack(jax.random.split(seed, n_samples))

        params_final_batch, log_det_batch = jax.vmap(
            lambda tok, rng_: solve_augmented_ode(
                constant_vf,
                tok,
                solver,
                rng_,
                rtol=1e-5,
                atol=1e-5,
                n_epsilon=20,
            )
        )(tokens, rngs)

        # Compute log probs: log_p(z_0) - log_det_jacobian
        log_p_z0_batch = norm.logpdf(params_final_batch.data[:, tokens.partition_idx, 0])
        log_probs = log_p_z0_batch - log_det_batch

        # Compare with expected (stricter tolerances)
        assert jnp.allclose(
            log_probs,
            expected_log_probs,
            rtol=0.05,
            atol=0.1,
        )

    def test_doubling_vf_log_prob(self, solver):
        """Test log prob computation for doubling VF.

        For f(x) = log(2)·x, the flow scales samples by 2.
        Sample density changes from N(0,1) to N(0,4).
        """
        def doubling_vf(tok, t):
            return jnp.log(2.0) * tok.data[:, tok.partition_idx:]

        # Create 100 sample Tokens from N(0, 1), n_tokens=1, batch=1
        seed = jax.random.PRNGKey(42)
        samples = jax.random.normal(seed, (100, 1, 1))
        n_samples = 100
        rngs = jnp.stack(jax.random.split(seed, n_samples))

        # Expected log prob for transformed samples at N(0, 2)
        expected_log_probs = norm.logpdf(samples[:, 0, 0], scale=2.0)

        tokens = create_mock_tokens(
            jnp.zeros((n_samples, 1, 1)),
            samples,
            sample_ndims=1
        )

        params_final_batch, log_det_batch = jax.vmap(
            lambda tok, rng_: solve_augmented_ode(
                doubling_vf,
                tok,
                solver,
                rng_,
                rtol=1e-5,
                atol=1e-5,
                n_epsilon=30,
            )
        )(tokens, rngs)

        # Compute log probs: log_p(z_0) - log_det_jacobian
        log_p_z0_batch = norm.logpdf(params_final_batch.data[:, tokens.partition_idx, 0])
        log_probs = log_p_z0_batch - log_det_batch

        # Compare with expected (stochastic trace estimation is noisy)
        # Max relative error ~34% due to trace estimation variance
        assert jnp.allclose(
            log_probs,
            expected_log_probs,
            rtol=0.2,
            atol=0.3,
        )

    def test_augmented_state_shape(self, solver):
        """Test that augmented ODE returns correct output shape."""
        def vf(tok, t):
            return -tok.data[:, tok.partition_idx:]  # Simple decay

        x0_data = jnp.array([[1.0, 2.0, 3.0]])
        tokens = create_mock_tokens(
            jnp.zeros((1, 1)),
            x0_data
        )
        rng = jax.random.PRNGKey(0)

        params_final, log_det = solve_augmented_ode(
            vf,
            tokens,
            solver,
            rng,
        )

        assert params_final.data.shape == tokens.data.shape
        assert log_det.shape == ()  # Scalar
