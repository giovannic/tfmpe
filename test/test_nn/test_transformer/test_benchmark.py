"""Benchmark tests for Transformer memory and speed."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from tfmpe.nn.transformer import Transformer
from tfmpe.nn.transformer.config import TransformerConfig
from tfmpe.preprocessing import Labeller, Tokens

def create_benchmark_tokens(batch_size: int, seq_len: int) -> Tokens:
    """Create tokens for benchmarking.

    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Number of parameter tokens

    Returns
    -------
    Tokens
        Tokens object for benchmarking
    """
    labeller = Labeller(label_map={'context': 0, 'param': 1})

    data = {
        'context': jnp.ones((batch_size, 2, 1)),
        'param': jnp.ones((batch_size, seq_len, 1)),
    }

    return Tokens.from_pytree(
        data,
        labeller=labeller,
        condition=['context'],
        sample_ndims=1,
    )


BATCH_SIZE = 32
SEQ_LENS = [10, 50, 100, 200, 500]
N_WARMUP = 3


def _make_transformer(tokens):
    """Create a Transformer and common inputs for benchmarking."""
    config = TransformerConfig(
        latent_dim=128,
        n_encoder=2,
        n_heads=4,
        n_ff=2,
        label_dim=32,
    )
    rngs = nnx.Rngs(0)
    transformer = Transformer(config=config, tokens=tokens, rngs=rngs)
    time_input = jnp.ones((BATCH_SIZE,))
    return transformer, time_input


class TestTransformerBenchmark:
    """Memory and speed benchmarks for Transformer forward/backward pass."""

    # ── Timing (pytest-benchmark) ──────────────────────────────

    @pytest.mark.slow
    @pytest.mark.benchmark(group="forward")
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_benchmark(self, benchmark, seq_len: int) -> None:
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens)

        @nnx.jit
        def forward_fn(model, tokens, t):
            return model(tokens=tokens, time=t, deterministic=True)

        # Pre-compile
        forward_fn(transformer, tokens, time_input).block_until_ready()

        def run():
            result = forward_fn(transformer, tokens, time_input)
            result.block_until_ready()
            return result

        benchmark.pedantic(run, warmup_rounds=N_WARMUP, rounds=10, iterations=1)

    @pytest.mark.slow
    @pytest.mark.benchmark(group="backward")
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_backward_benchmark(self, benchmark, seq_len: int) -> None:
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens)

        def loss_fn(model):
            output = model(tokens=tokens, time=time_input, deterministic=True)
            return jnp.mean(output ** 2)

        grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

        # Pre-compile
        loss, _ = grad_fn(transformer)
        loss.block_until_ready()

        def run():
            loss, grads = grad_fn(transformer)
            loss.block_until_ready()
            return loss

        benchmark.pedantic(run, warmup_rounds=N_WARMUP, rounds=10, iterations=1)

    # ── Memory (hand-rolled) ───────────────────────────────────

    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_forward_peak_memory(self, seq_len: int) -> None:
        device = jax.local_devices()[0]
        if not hasattr(device, "memory_stats"):
            pytest.skip("Device does not support memory_stats")

        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens)

        @nnx.jit
        def forward_fn(model, tokens, t):
            return model(tokens=tokens, time=t, deterministic=True)

        # Warmup / compile
        for _ in range(N_WARMUP):
            forward_fn(transformer, tokens, time_input).block_until_ready()

        _ = device.memory_stats()  # reset baseline
        forward_fn(transformer, tokens, time_input).block_until_ready()

        stats = device.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peak_mb = stats["peak_bytes_in_use"] / (1024 * 1024)
            print(f"\nForward peak memory (seq_len={seq_len}): {peak_mb:.2f} MB")

    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_backward_peak_memory(self, seq_len: int) -> None:
        device = jax.local_devices()[0]
        if not hasattr(device, "memory_stats"):
            pytest.skip("Device does not support memory_stats")

        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens)

        def loss_fn(model):
            output = model(tokens=tokens, time=time_input, deterministic=True)
            return jnp.mean(output ** 2)

        grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

        # Warmup / compile
        for _ in range(N_WARMUP):
            loss, _ = grad_fn(transformer)
            loss.block_until_ready()

        _ = device.memory_stats()  # reset baseline
        loss, _ = grad_fn(transformer)
        loss.block_until_ready()

        stats = device.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peak_mb = stats["peak_bytes_in_use"] / (1024 * 1024)
            print(f"\nBackward peak memory (seq_len={seq_len}): {peak_mb:.2f} MB")
