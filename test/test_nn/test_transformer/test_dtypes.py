"""Tests for mixed precision dtype propagation."""

import pytest
import jax.numpy as jnp
from flax import nnx

from tfmpe.nn.transformer import Transformer
from tfmpe.nn.transformer.config import TransformerConfig

PRECISION_MODES = {
    "float32": dict(ops_dtype=jnp.float32, sensitive_ops_dtype=jnp.float32),
    "bfloat16": dict(ops_dtype=jnp.bfloat16, sensitive_ops_dtype=jnp.bfloat16),
    "mixed": dict(ops_dtype=jnp.bfloat16, sensitive_ops_dtype=jnp.float32),
}


def _make_transformer(tokens, precision_mode="float32"):
    config = TransformerConfig(
        latent_dim=64,
        n_encoder=2,
        n_heads=4,
        n_ff=2,
        label_dim=16,
        **PRECISION_MODES[precision_mode],
    )
    rngs = nnx.Rngs(0)
    return Transformer(config=config, tokens=tokens, rngs=rngs)


class TestOutputDtype:
    """Verify the output dtype of the transformer for each precision mode."""

    @pytest.mark.parametrize("precision_mode,expected_dtype", [
        ("float32", jnp.float32),
        ("bfloat16", jnp.bfloat16),
        ("mixed", jnp.bfloat16),
    ])
    def test_forward_output_dtype(
        self, simple_tokens, precision_mode, expected_dtype
    ):
        transformer = _make_transformer(simple_tokens, precision_mode)
        time = jnp.array(0.5)
        output = transformer(tokens=simple_tokens, time=time)
        assert output.dtype == expected_dtype

    @pytest.mark.parametrize("precision_mode,expected_dtype", [
        ("float32", jnp.float32),
        ("bfloat16", jnp.bfloat16),
        ("mixed", jnp.bfloat16),
    ])
    def test_encode_output_dtype(
        self, simple_tokens, precision_mode, expected_dtype
    ):
        transformer = _make_transformer(simple_tokens, precision_mode)
        time = jnp.array(0.5)
        encoded = transformer.encode(tokens=simple_tokens, time=time)
        assert encoded.dtype == expected_dtype


class TestEmbeddingDtype:
    """Verify embedding layer output dtype matches ops_dtype."""

    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_embedding_output_dtype(self, simple_tokens, precision_mode):
        transformer = _make_transformer(simple_tokens, precision_mode)
        expected = PRECISION_MODES[precision_mode]["ops_dtype"]
        time = jnp.array(0.5)
        embedded = transformer.embedding(simple_tokens, time)
        assert embedded.dtype == expected


class TestOutputShapePreserved:
    """Verify output shapes are unchanged across precision modes."""

    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_output_shape(self, simple_tokens, precision_mode):
        transformer = _make_transformer(simple_tokens, precision_mode)
        time = jnp.array(0.5)
        output = transformer(tokens=simple_tokens, time=time)
        assert output.shape == (1, 3, 1)
