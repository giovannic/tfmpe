"""Tests for MLP model."""

import diffrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tfmpe.nn.mlp import MLP
from tfmpe.preprocessing import Tokens
from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution


@pytest.fixture
def simple_tokens() -> Tokens:
    """Create simple tokens with obs and theta."""
    data = {
        'obs': jnp.ones((1, 3, 2)),   # 3 obs tokens, value_dim=2
        'theta': jnp.ones((1, 2, 2))  # 2 param tokens
    }
    return Tokens.from_pytree(
        data, condition=['obs'], sample_ndims=1, pad_to_even=False
    )


@pytest.fixture
def mlp(simple_tokens: Tokens) -> MLP:
    """Create MLP for testing."""
    rngs = nnx.Rngs(0)
    return MLP(n_ff=2, latent_dim=32, tokens=simple_tokens, rngs=rngs)


class TestMLPScalarTime:
    """Tests for MLP with scalar time (ODE solver case)."""

    def test_output_shape_scalar_time(
        self,
        mlp: MLP,
        simple_tokens: Tokens,
    ) -> None:
        """Test MLP output shape with scalar time.

        Given:
        - MLP initialized with tokens
        - Scalar time value (as passed by ODE solver)

        When:
        - Forward pass through MLP

        Then:
        - Output shape is (*sample_shape, n_target_tokens, value_dim)
        - All values are finite
        """
        time = jnp.array(0.5)
        output = mlp(simple_tokens, time)

        # n_target_tokens = 2 (theta), value_dim = 2
        expected_shape = (1, 2, 2)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output))


class TestMLPBatchedTime:
    """Tests for MLP with batched time."""

    def test_output_shape_batched_time(
        self,
        mlp: MLP,
    ) -> None:
        """Test MLP output shape with batched samples and time.

        Given:
        - MLP initialized with tokens
        - Batched tokens with sample_shape=(4,)
        - Batched time with shape (4,)

        When:
        - Forward pass through MLP

        Then:
        - Output shape is (4, n_target_tokens, value_dim)
        - All values are finite
        """
        batched_data = {
            'obs': jnp.ones((4, 3, 2)),
            'theta': jnp.ones((4, 2, 2))
        }
        batched_tokens = Tokens.from_pytree(
            batched_data, condition=['obs'], sample_ndims=1, pad_to_even=False
        )
        batched_time = jnp.ones((4,)) * 0.5

        output = mlp(batched_tokens, batched_time)

        expected_shape = (4, 2, 2)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output))


class TestMLPIntegration:
    """Integration tests for MLP with TFMPE.sample_posterior."""

    @pytest.fixture
    def tokens(self) -> Tokens:
        """Create tokens for integration test."""
        data = {
            'obs': jnp.ones((1, 3, 2)),
            'theta': jnp.ones((1, 2, 2))
        }
        return Tokens.from_pytree(
            data, condition=['obs'], sample_ndims=1, pad_to_even=False
        )

    @pytest.fixture
    def tfmpe_with_mlp(self, tokens: Tokens) -> TFMPE:
        """TFMPE instance with MLP vector field."""
        rngs = nnx.Rngs(
            params=jax.random.PRNGKey(0),
        )
        mlp = MLP(n_ff=2, latent_dim=32, tokens=tokens, rngs=rngs)
        base_dist = NormalDistribution(rngs=rngs)

        tfmpe = TFMPE(
            vf_network=mlp,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
            ode_kwargs={'rtol': 1e-3, 'atol': 1e-3},
        )
        tfmpe.eval()
        return tfmpe

    def test_sample_posterior_with_mlp(
        self,
        tfmpe_with_mlp: TFMPE,
        tokens: Tokens,
    ) -> None:
        """Test TFMPE.sample_posterior works with MLP vector field.

        Given:
        - TFMPE instance with MLP as vf_network
        - Tokens with context (obs) and params (theta)

        When:
        - Call sample_posterior()

        Then:
        - Output is a Tokens instance
        - Output shape matches input shape
        - All values are finite
        """
        samples = tfmpe_with_mlp.sample_posterior(tokens)

        assert isinstance(samples, Tokens)
        assert samples.data.shape == tokens.data.shape
        assert jnp.all(jnp.isfinite(samples.data))

    def test_sample_posterior_batched(
        self,
        tfmpe_with_mlp: TFMPE,
    ) -> None:
        """Test TFMPE.sample_posterior with batched samples using MLP.

        Given:
        - TFMPE instance with MLP
        - Batched tokens with sample_shape=(10,)

        When:
        - Call sample_posterior()

        Then:
        - Output shape matches input (10, n_tokens, value_dim)
        - All samples are finite
        - Samples differ (not deterministic across batch)
        """
        batched_data = {
            'obs': jnp.ones((10, 3, 2)),
            'theta': jnp.ones((10, 2, 2))
        }
        batched_tokens = Tokens.from_pytree(
            batched_data, condition=['obs'], sample_ndims=1, pad_to_even=False
        )

        samples = tfmpe_with_mlp.sample_posterior(batched_tokens)

        assert samples.data.shape == batched_tokens.data.shape
        assert jnp.all(jnp.isfinite(samples.data))
        # Target data (theta) should differ across batch dimension
        target_data = samples.data[:, samples.partition_idx:]
        assert not jnp.allclose(target_data[0], target_data[1])
