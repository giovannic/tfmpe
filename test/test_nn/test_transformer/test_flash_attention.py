"""Tests for Flash Attention v4 CuTeDSL kernel.

These tests require:
- A Blackwell GPU (SM100+ / SM120)
- nvidia-cutlass-dsl >= 4.2.0, jax-tvm-ffi, flash_attn.cute

All tests are skipped when the hardware or dependencies are unavailable.
"""

import math

import pytest
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_SKIP_REASON_DEPS = "flash_attention_v4 dependencies not installed"
_SKIP_REASON_GPU = "No Blackwell GPU available (requires SM100+)"

try:
    from tfmpe.nn.transformer.flash_attention import (
        flash_attention_v4,
        register_flash_attn_ops,
        _get_arch,
        _mask_to_seqlens,
        _pack_varlen,
        _unpack_varlen,
        _REGISTERED,
    )
    _HAS_DEPS = True
except (ImportError, ModuleNotFoundError):
    _HAS_DEPS = False


def _has_blackwell_gpu() -> bool:
    if not _HAS_DEPS:
        return False
    try:
        arch = _get_arch()
        return arch // 10 in (10, 11, 12)
    except Exception:
        return False


requires_deps = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON_DEPS)
requires_blackwell = pytest.mark.skipif(
    not _has_blackwell_gpu(), reason=_SKIP_REASON_GPU,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[jnp.bfloat16, jnp.float16])
def dtype(request):
    return request.param


@pytest.fixture(params=[(64, 4), (128, 8)])
def head_config(request):
    """(head_dim, num_heads)"""
    return request.param


# ---------------------------------------------------------------------------
# Unit tests: mask / varlen utilities (CPU-only, no GPU needed)
# ---------------------------------------------------------------------------


@requires_deps
class TestMaskUtilities:
    """Tests for mask → cu_seqlens conversion and packing."""

    def test_mask_to_seqlens_full_mask(self):
        """All-True mask yields full sequence lengths."""
        mask = jnp.ones((2, 4, 8, 8), dtype=jnp.bool_)  # (B, H, Q, K)
        sq, sk = _mask_to_seqlens(mask)
        assert jnp.all(sq == 8)
        assert jnp.all(sk == 8)

    def test_mask_to_seqlens_partial_mask(self):
        """Partial mask yields correct per-batch lengths."""
        # batch=2, heads=1, q=4, k=4
        mask = jnp.ones((2, 1, 4, 4), dtype=jnp.bool_)
        # Zero out last 2 positions for batch item 1
        mask = mask.at[1, :, 2:, :].set(False)
        mask = mask.at[1, :, :, 2:].set(False)

        sq, sk = _mask_to_seqlens(mask)
        assert int(sq[0]) == 4
        assert int(sq[1]) == 2
        assert int(sk[0]) == 4
        assert int(sk[1]) == 2

    def test_pack_varlen_shapes(self):
        """Packing produces correct shapes and cu_seqlens."""
        batch, sq, sk, h, d = 3, 8, 8, 4, 64
        q = jnp.ones((batch, sq, h, d))
        k = jnp.ones((batch, sk, h, d))
        v = jnp.ones((batch, sk, h, d))
        seqlens_q = jnp.array([4, 8, 6], dtype=jnp.int32)
        seqlens_k = jnp.array([4, 8, 6], dtype=jnp.int32)

        qp, kp, vp, cu_q, cu_k = _pack_varlen(q, k, v, seqlens_q, seqlens_k)

        assert qp.shape == (batch * sq, h, d)
        assert kp.shape == (batch * sk, h, d)
        assert cu_q.shape == (batch + 1,)
        assert int(cu_q[0]) == 0
        assert int(cu_q[-1]) == 4 + 8 + 6

    def test_unpack_roundtrip(self):
        """Pack → FFI output → unpack preserves shape."""
        batch, sq, h, dv = 2, 16, 4, 64
        out_packed = jnp.ones((batch * sq, h, dv))
        out = _unpack_varlen(out_packed, batch, sq, h, dv)
        assert out.shape == (batch, sq, h, dv)


# ---------------------------------------------------------------------------
# Forward pass tests (require Blackwell GPU)
# ---------------------------------------------------------------------------


@requires_deps
@requires_blackwell
class TestFlashAttentionForward:
    """Forward-pass correctness tests against JAX reference."""

    def test_registration_caches(self, head_config, dtype):
        """Calling register twice with same config is a no-op."""
        head_dim, num_heads = head_config
        _REGISTERED.clear()
        register_flash_attn_ops(
            head_dim=head_dim, num_heads=num_heads, dtype=dtype,
        )
        n_before = len(_REGISTERED)
        register_flash_attn_ops(
            head_dim=head_dim, num_heads=num_heads, dtype=dtype,
        )
        assert len(_REGISTERED) == n_before

    def test_forward_output_shape(self, head_config, dtype):
        """Output shape matches expectation."""
        head_dim, num_heads = head_config
        batch, seqlen = 2, 64
        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, num_heads, head_dim),
            dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, num_heads, head_dim),
            dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, num_heads, head_dim),
            dtype=dtype,
        )

        out = flash_attention_v4(q, k, v)
        assert out.shape == (batch, seqlen, num_heads, head_dim)
        assert out.dtype == dtype

    def test_forward_matches_reference(self):
        """Output is close to JAX dot_product_attention reference."""
        batch, seqlen, heads, dim = 2, 128, 4, 64
        dtype = jnp.bfloat16

        key = jax.random.key(42)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (batch, seqlen, heads, dim), dtype=dtype)
        k = jax.random.normal(k2, (batch, seqlen, heads, dim), dtype=dtype)
        v = jax.random.normal(k3, (batch, seqlen, heads, dim), dtype=dtype)

        out = flash_attention_v4(q, k, v)
        ref = jax.nn.dot_product_attention(q, k, v)

        max_diff = float(jnp.abs(out - ref).max())
        assert max_diff < 1e-2, (
            f"Forward max diff {max_diff:.4e} exceeds tolerance"
        )

    def test_forward_jit(self):
        """flash_attention_v4 works under jax.jit."""
        batch, seqlen, heads, dim = 2, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        out_eager = flash_attention_v4(q, k, v)
        out_jit = jax.jit(flash_attention_v4)(q, k, v)

        max_diff = float(jnp.abs(out_eager - out_jit).max())
        assert max_diff < 1e-5, f"JIT vs eager diff: {max_diff:.4e}"

    def test_forward_deterministic(self):
        """Same inputs produce identical outputs."""
        batch, seqlen, heads, dim = 1, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        out1 = flash_attention_v4(q, k, v)
        out2 = flash_attention_v4(q, k, v)
        assert jnp.allclose(out1, out2), "Non-deterministic forward pass"

    @pytest.mark.parametrize("seqlen", [32, 64, 128, 256])
    def test_forward_various_seqlens(self, seqlen):
        """Forward works across sequence lengths."""
        batch, heads, dim = 2, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        out = flash_attention_v4(q, k, v)
        assert out.shape == (batch, seqlen, heads, dim)
        assert jnp.all(jnp.isfinite(out)), "Output contains NaN/Inf"

    def test_forward_with_batch_dims(self):
        """Handles extra leading batch dimensions."""
        heads, dim = 4, 64
        dtype = jnp.bfloat16
        # shape: (samples, batch, seqlen, heads, dim)
        q = jax.random.normal(
            jax.random.key(0), (3, 2, 64, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (3, 2, 64, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (3, 2, 64, heads, dim), dtype=dtype,
        )

        out = flash_attention_v4(q, k, v)
        assert out.shape == (3, 2, 64, heads, dim)

    def test_forward_dtype_cast(self):
        """Output dtype respects the `dtype` kwarg."""
        batch, seqlen, heads, dim = 1, 64, 4, 64
        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=jnp.bfloat16,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=jnp.bfloat16,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=jnp.bfloat16,
        )

        out = flash_attention_v4(q, k, v, dtype=jnp.float32)
        assert out.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Backward pass tests (require Blackwell GPU)
# ---------------------------------------------------------------------------


@requires_deps
@requires_blackwell
class TestFlashAttentionBackward:
    """Backward-pass (gradient) tests."""

    def test_grad_runs(self):
        """jax.grad through flash_attention_v4 produces finite grads."""
        batch, seqlen, heads, dim = 2, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        def loss_fn(q, k, v):
            return flash_attention_v4(q, k, v).sum()

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert jnp.all(jnp.isfinite(dq)), "dq contains NaN/Inf"
        assert jnp.all(jnp.isfinite(dk)), "dk contains NaN/Inf"
        assert jnp.all(jnp.isfinite(dv)), "dv contains NaN/Inf"

    def test_grad_matches_reference(self):
        """Gradients are close to JAX reference implementation."""
        batch, seqlen, heads, dim = 1, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        def flash_loss(q, k, v):
            return flash_attention_v4(q, k, v).sum()

        def ref_loss(q, k, v):
            return jax.nn.dot_product_attention(q, k, v).sum()

        dq, dk, dv = jax.grad(flash_loss, argnums=(0, 1, 2))(q, k, v)
        dq_ref, dk_ref, dv_ref = jax.grad(ref_loss, argnums=(0, 1, 2))(
            q, k, v,
        )

        # bf16 gradients have looser tolerance
        for name, got, ref in [
            ("dq", dq, dq_ref),
            ("dk", dk, dk_ref),
            ("dv", dv, dv_ref),
        ]:
            got_f32 = got.astype(jnp.float32)
            ref_f32 = ref.astype(jnp.float32)
            max_diff = float(jnp.abs(got_f32 - ref_f32).max())
            assert max_diff < 5e-2, (
                f"{name} grad max diff {max_diff:.4e} exceeds tolerance"
            )

    def test_grad_jit(self):
        """Gradients work under jax.jit."""
        batch, seqlen, heads, dim = 1, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        def loss_fn(q, k, v):
            return flash_attention_v4(q, k, v).sum()

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
        dq, dk, dv = grad_fn(q, k, v)

        assert jnp.all(jnp.isfinite(dq))
        assert jnp.all(jnp.isfinite(dk))
        assert jnp.all(jnp.isfinite(dv))


# ---------------------------------------------------------------------------
# Varlen / mask tests (require Blackwell GPU)
# ---------------------------------------------------------------------------


@requires_deps
@requires_blackwell
class TestFlashAttentionVarlen:
    """Tests for the varlen (padding mask → cu_seqlens) path."""

    def test_forward_with_mask(self):
        """Forward with a mask produces correct shape and finite values."""
        batch, seqlen, heads, dim = 2, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        # Padding mask: batch 0 has 48 valid tokens, batch 1 has 64
        mask = jnp.ones((batch, heads, seqlen, seqlen), dtype=jnp.bool_)
        mask = mask.at[0, :, 48:, :].set(False)
        mask = mask.at[0, :, :, 48:].set(False)

        out = flash_attention_v4(q, k, v, mask=mask)
        assert out.shape == (batch, seqlen, heads, dim)
        assert jnp.all(jnp.isfinite(out)), "Varlen output has NaN/Inf"

    def test_no_mask_vs_full_mask_equivalent(self):
        """All-True mask should produce same result as no mask."""
        batch, seqlen, heads, dim = 1, 64, 4, 64
        dtype = jnp.bfloat16

        q = jax.random.normal(
            jax.random.key(0), (batch, seqlen, heads, dim), dtype=dtype,
        )
        k = jax.random.normal(
            jax.random.key(1), (batch, seqlen, heads, dim), dtype=dtype,
        )
        v = jax.random.normal(
            jax.random.key(2), (batch, seqlen, heads, dim), dtype=dtype,
        )

        out_no_mask = flash_attention_v4(q, k, v)
        full_mask = jnp.ones(
            (batch, heads, seqlen, seqlen), dtype=jnp.bool_,
        )
        out_full_mask = flash_attention_v4(q, k, v, mask=full_mask)

        max_diff = float(jnp.abs(out_no_mask - out_full_mask).max())
        assert max_diff < 1e-2, (
            f"Full mask vs no mask diff: {max_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# Integration test: TransformerConfig dispatch
# ---------------------------------------------------------------------------


@requires_deps
@requires_blackwell
class TestFlashAttentionConfigIntegration:
    """Test that 'flashattention_v4' works through TransformerConfig."""

    def test_config_accepts_literal(self):
        """TransformerConfig accepts 'flashattention_v4'."""
        from tfmpe.nn.transformer.config import TransformerConfig

        config = TransformerConfig(
            latent_dim=64,
            n_heads=4,
            attention='flashattention_v4',
        )
        assert config.attention == 'flashattention_v4'

    def test_encoder_block_init(self):
        """EncoderBlock initializes with flashattention_v4."""
        from tfmpe.nn.transformer.config import TransformerConfig
        from tfmpe.nn.transformer.encoder import EncoderBlock

        config = TransformerConfig(
            latent_dim=64,
            n_heads=4,
            n_ff=2,
            attention='flashattention_v4',
        )
        rngs = nnx.Rngs(0)
        block = EncoderBlock(config=config, rngs=rngs)
        assert block is not None

    def test_encoder_block_forward(self):
        """EncoderBlock forward pass with flashattention_v4."""
        from tfmpe.nn.transformer.config import TransformerConfig
        from tfmpe.nn.transformer.encoder import EncoderBlock
        from flax import nnx

        config = TransformerConfig(
            latent_dim=64,
            n_heads=4,
            n_ff=2,
            attention='flashattention_v4',
            ops_dtype=jnp.bfloat16,
            sensitive_ops_dtype=jnp.bfloat16,
        )
        rngs = nnx.Rngs(0)
        block = EncoderBlock(config=config, rngs=rngs)

        batch, seqlen = 2, 64
        x = jax.random.normal(
            jax.random.key(0),
            (batch, seqlen, config.latent_dim),
            dtype=jnp.bfloat16,
        )

        out = block(x, deterministic=True)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out)), "Encoder output has NaN/Inf"

    def test_encoder_block_backward(self):
        """EncoderBlock backward pass (gradient) with flashattention_v4."""
        from tfmpe.nn.transformer.config import TransformerConfig
        from tfmpe.nn.transformer.encoder import EncoderBlock
        from flax import nnx

        config = TransformerConfig(
            latent_dim=64,
            n_heads=4,
            n_ff=2,
            attention='flashattention_v4',
            ops_dtype=jnp.bfloat16,
            sensitive_ops_dtype=jnp.bfloat16,
        )
        rngs = nnx.Rngs(0)
        block = EncoderBlock(config=config, rngs=rngs)

        batch, seqlen = 2, 64
        x = jax.random.normal(
            jax.random.key(0),
            (batch, seqlen, config.latent_dim),
            dtype=jnp.bfloat16,
        )

        def loss_fn(x):
            return block(x, deterministic=True).sum()

        dx = jax.grad(loss_fn)(x)
        assert dx.shape == x.shape
        assert jnp.all(jnp.isfinite(dx)), "Encoder gradient has NaN/Inf"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@requires_deps
class TestFlashAttentionErrors:
    """Test error handling and edge cases."""

    def test_unsupported_arch_raises(self):
        """register_flash_attn_ops raises on non-Blackwell GPUs."""
        arch = _get_arch()
        if arch // 10 in (10, 11, 12):
            pytest.skip("Running on Blackwell — cannot test arch error")

        with pytest.raises(ValueError, match="requires Blackwell"):
            register_flash_attn_ops(head_dim=64, num_heads=4)
