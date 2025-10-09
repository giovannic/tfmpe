"""Tests for transformer encoder components (MLP)."""

import pytest
import jax.numpy as jnp
from flax import nnx

from tfmpe.nn.transformer.config import TransformerConfig
from tfmpe.nn.transformer.encoder import MLP


class TestMLP:
    """Tests for MLP feedforward network."""

    @pytest.mark.parametrize("n_ff", [1, 2, 4, 8])
    def test_output_shape_preservation(self, n_ff: int) -> None:
        """Test that MLP preserves input shape.

        Parameters
        ----------
        n_ff : int
            Number of feedforward layers
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=128,
            n_encoder=4,
            n_heads=8,
            n_ff=n_ff,
            label_dim=32,
            index_out_dim=64,
            dropout=0.1,
            ops_dtype=jnp.float32,
            sensitive_ops_dtype=jnp.float32,
        )

        mlp = MLP(config=config, rngs=rngs)

        batch_size, n_tokens = 4, 10
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (batch_size, n_tokens, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Test MLP with different batch sizes.

        Parameters
        ----------
        batch_size : int
            Number of samples in batch
        """
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
            ops_dtype=jnp.float32,
            sensitive_ops_dtype=jnp.float32,
        )

        mlp = MLP(config=config, rngs=rngs)

        n_tokens = 20
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (batch_size, n_tokens, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_train_eval_dropout_difference(self) -> None:
        """Test that dropout behaves differently in train vs eval mode."""
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=128,
            n_encoder=4,
            n_heads=8,
            n_ff=2,
            label_dim=32,
            index_out_dim=64,
            dropout=0.5,
            ops_dtype=jnp.float32,
            sensitive_ops_dtype=jnp.float32,
        )

        mlp = MLP(config=config, rngs=rngs)

        batch_size, n_tokens = 4, 10
        inputs = jnp.ones((batch_size, n_tokens, config.latent_dim))

        # Set to eval mode
        mlp.eval()
        eval_output = mlp(inputs)

        # In eval mode, output should be deterministic
        eval_output_2 = mlp(inputs)
        assert jnp.allclose(eval_output, eval_output_2), (
            "Eval mode should produce deterministic outputs"
        )

    def test_with_multidimensional_batch(self) -> None:
        """Test MLP with multiple sample dimensions."""
        rngs = nnx.Rngs(0)
        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=4,
            n_ff=2,
            label_dim=16,
            index_out_dim=32,
            dropout=0.1,
            ops_dtype=jnp.float32,
            sensitive_ops_dtype=jnp.float32,
        )

        mlp = MLP(config=config, rngs=rngs)

        # Shape: (samples, batch, n_tokens, latent_dim)
        inputs = jnp.ones((2, 3, 10, config.latent_dim))
        output = mlp(inputs)

        expected_shape = (2, 3, 10, config.latent_dim)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

