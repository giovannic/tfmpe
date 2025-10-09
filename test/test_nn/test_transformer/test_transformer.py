"""Tests for Transformer model end-to-end functionality."""

import pytest
import jax.numpy as jnp
from flax import nnx

from tfmpe.preprocessing import Tokens, Labeller
from tfmpe.nn.transformer import Transformer
from tfmpe.nn.transformer.config import TransformerConfig

class TestTransformerInit:
    """Tests for Transformer initialization."""

    def test_init_accepts_config_dataclass(
        self,
        simple_tokens: Tokens,
    ) -> None:
        """Test that Transformer.__init__ accepts config dataclass."""
        config = TransformerConfig(
            latent_dim=128,
            n_encoder=2,
            n_heads=8,
            n_ff=2,
            label_dim=32,
            index_out_dim=64,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)

        # Should accept config as primary parameter
        transformer = Transformer(
            config=config,
            tokens=simple_tokens,
            rngs=rngs,
        )

        assert transformer is not None


class TestTransformerForwardPass:
    """Tests for Transformer forward pass."""

    def test_forward_pass_output_shape(
        self,
        simple_tokens: Tokens,
    ) -> None:
        """Test forward pass output shape matches input param
        shape."""
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)

        transformer = Transformer(
            config=config,
            tokens=simple_tokens,
            rngs=rngs,
        )

        # Time is a scalar for now
        time = jnp.array(0.5)

        # Forward pass using __call__
        output = transformer(
            tokens=simple_tokens,
            time=time,
        )

        expected_shape = (1, 3, 1)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    @pytest.mark.parametrize(
        "n_param_tokens,latent_dim",
        [
            (2, 64),
            (5, 128),
        ],
    )
    def test_forward_pass_various_sizes(
        self,
        n_param_tokens: int,
        latent_dim: int,
    ) -> None:
        """Test forward pass with various token and latent
        dimensions."""
        # Create labeller for this test
        labeller = Labeller(label_map={'context': 0, 'param': 1})

        # Create context and param tokens
        data = {
            'context': jnp.ones((1, 2, 1)),
            'param': jnp.ones((1, n_param_tokens, 1))
        }

        tokens = Tokens.from_pytree(
            data,
            labeller=labeller,
            condition=['context'],
            sample_ndims=1,
            pad_to_even=False,
        )

        config = TransformerConfig(
            latent_dim=latent_dim,
            n_encoder=2,
            n_heads=min(4, latent_dim // 16),
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config,
            tokens=tokens,
            rngs=rngs,
        )

        time = jnp.array(0.5)
        output = transformer(
            tokens=tokens,
            time=time,
        )

        # Expected shape: param is (n_param_tokens, 1)
        expected_shape = (1, n_param_tokens, 1)
        assert output.shape == expected_shape


class TestTransformerEncode:
    """Tests for Transformer.encode method."""

    def test_encode_method_output_shape(
        self,
        simple_tokens: Tokens,
    ) -> None:
        """Test encode method returns correct shape."""
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)

        transformer = Transformer(
            config=config,
            tokens=simple_tokens,
            rngs=rngs,
        )

        time = jnp.array(0.5)

        # Encode context tokens
        encoded = transformer.encode(
            tokens=simple_tokens,
            time=time,
        )

        # Expected shape: (3 tokens total for mu + obs, latent_dim)
        expected_shape = (1, 3, config.latent_dim)
        assert encoded.shape == expected_shape

class TestTransformerSampleDimensions:
    """Tests for Transformer with multiple sample dimensions."""

    def test_sample_same_sample_dimensions(self) -> None:
        """Test __call__ with same sample dimensions for both
        tokens."""
        # Create labeller for this test
        labeller = Labeller(label_map={'context': 0, 'param': 1})

        # Create context and param tokens
        tokens = Tokens.from_pytree(
            {
                'param': jnp.ones((2, 4, 5, 1)),
                'context': jnp.ones((2, 4, 3, 1))
            },
            condition=['context'],
            labeller=labeller,
            sample_ndims=2,
            batch_ndims={'context': 1},
            pad_to_even=False,
        )

        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
        )

        rngs = nnx.Rngs(0)
        transformer = Transformer(
            config=config,
            tokens=tokens,
            rngs=rngs,
        )

        time = jnp.ones((2, 4))

        output = transformer(
            tokens=tokens,
            time=time,
        )

        # Expected shape: param is (2, 4, 5, 1)
        expected_shape = (2, 4, 5, 1)
        assert output.shape == expected_shape
