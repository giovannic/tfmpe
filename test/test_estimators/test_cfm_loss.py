"""Unit tests for CFM loss function."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import _cfm_loss as cfm_loss
from tfmpe.preprocessing.tokens import Tokens
from jaxtyping import Array


class SimpleVectorField:
    """Simple deterministic vector field for testing."""

    def __call__(self, tokens: Tokens,
                 t: Array) -> Array:
        """Return scaled params as velocity."""
        return jnp.tanh(tokens.data[:, tokens.partition_idx:] / 10.0)


class TestCFMLossBatched:
    """Test CFM loss function with batched inputs."""

    @pytest.fixture
    def vf_network(self):
        """Simple vector field for testing."""
        return SimpleVectorField()

    @pytest.fixture
    def tfmpe_instance(self, vf_network):
        """TFMPE instance with simple vector field."""
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        base_dist = NormalDistribution(rngs=rngs)
        return TFMPE(
            vf_network=vf_network,
            base_dist=base_dist,
        )

    def test_cfm_loss_batched_returns_scalar(
        self, tfmpe_instance: TFMPE
    ) -> None:
        """Test cfm_loss with batched inputs returns scalar.

        Given:
        - TFMPE instance
        - Batched theta and context Tokens (batch_size=4)
        - Batched times with shape (4,)

        When:
        - Compute cfm_loss()

        Then:
        - Loss is a scalar (shape ())
        - Loss is finite
        """
        batch_size = 4
        state_dim = 10
        context_dim = 5

        # Create batched tokens
        theta_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, state_dim)
        )
        context_data = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, context_dim)
        )
        tokens = Tokens.from_pytree(
            {
                'x': theta_data,
                'y': context_data
            },
            condition=['y'],
            sample_ndims=1,
        )

        losses = cfm_loss(
            tfmpe_instance,
            tokens,
            jax.random.key(0)
        )

        # Check per-sample losses shape
        assert len(losses.shape) == 0

        # Check all finite
        assert jnp.all(jnp.isfinite(losses)), (
            "Losses contain NaN or Inf"
        )

    def test_cfm_loss_is_non_negative(
        self, tfmpe_instance: TFMPE
    ) -> None:
        """Test CFM loss is non-negative (MSE property).

        Given:
        - TFMPE instance with batched data
        - Multiple random times

        When:
        - Compute loss for different batch configurations

        Then:
        - All losses are non-negative
        """
        batch_size = 8
        state_dim = 10
        context_dim = 5

        # Create batched data
        theta_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, state_dim)
        )
        context_data = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, context_dim)
        )
        tokens = Tokens.from_pytree(
            {
                'x': theta_data,
                'y': context_data
            },
            condition=['y'],
            sample_ndims=1,
        )

        # Test multiple random time batches
        for i in range(5):
            losses = cfm_loss(
                tfmpe_instance,
                tokens,
                jax.random.key(i)
            )
            assert jnp.all(losses >= 0), (
                f"Losses are negative at iteration {i}: {losses}"
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_cfm_loss_different_batch_sizes(
        self, tfmpe_instance: TFMPE, batch_size: int
    ) -> None:
        """Test cfm_loss works with different batch sizes.

        Given:
        - TFMPE instance
        - batch_size in [1, 2, 8, 16]

        When:
        - Compute loss with batched inputs

        Then:
        - Succeeds and produces scalar output
        """
        state_dim = 10
        context_dim = 5

        # Create batched data
        theta_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, state_dim)
        )
        context_data = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, context_dim)
        )
        tokens = Tokens.from_pytree(
            {
                'x': theta_data,
                'y': context_data
            },
            condition=['y'],
            sample_ndims=1,
        )

        losses = cfm_loss(
            tfmpe_instance,
            tokens,
            jax.random.key(0)
        )

        assert len(losses.shape) == 0
        assert jnp.all(jnp.isfinite(losses))
