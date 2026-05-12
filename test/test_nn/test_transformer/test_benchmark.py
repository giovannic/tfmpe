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


BATCH_SIZE = 100
SEQ_LENS = [10, 50, 100, 200, 500, 1_000, 10_000]
N_WARMUP = 3

PRECISION_MODES = {
    "float32": dict(ops_dtype=jnp.float32, sensitive_ops_dtype=jnp.float32),
    "bfloat16": dict(ops_dtype=jnp.bfloat16, sensitive_ops_dtype=jnp.bfloat16),
    "mixed": dict(ops_dtype=jnp.bfloat16, sensitive_ops_dtype=jnp.float32),
}


def _make_transformer(tokens, precision_mode="float32"):
    """Create a Transformer and common inputs for benchmarking."""
    config = TransformerConfig(
        latent_dim=256,
        n_encoder=2,
        n_heads=4,
        n_ff=2,
        label_dim=8,
        max_positions=128,
        max_groups=128,
        attention='cudnn',
        **PRECISION_MODES[precision_mode],
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
    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_forward_benchmark(self, benchmark, seq_len: int, precision_mode: str) -> None:
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens, precision_mode)

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
    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_backward_benchmark(self, benchmark, seq_len: int, precision_mode: str) -> None:
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens, precision_mode)

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

    # ── Memory (peak usage via device.memory_stats) ────────────
    #
    # Reports peak_bytes_in_use after warmup+run for each config.
    # Since peak is monotonic within a process, compare across
    # precision modes at the same seq_len by running them in
    # separate pytest-xdist workers or sequential invocations.

    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_forward_peak_memory(self, seq_len: int, precision_mode: str) -> None:
        device = jax.local_devices()[0]
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens, precision_mode)

        @nnx.jit
        def forward_fn(model, tokens, t):
            return model(tokens=tokens, time=t, deterministic=True)

        # Warmup / compile
        for _ in range(N_WARMUP):
            forward_fn(transformer, tokens, time_input).block_until_ready()

        stats = device.memory_stats()
        if stats is None:
            pytest.skip("memory_stats() unavailable (platform allocator?)")

        peak_mb = stats["peak_bytes_in_use"] / (1024 * 1024)
        print(f"\nForward peak memory (seq_len={seq_len}, {precision_mode}): {peak_mb:.2f} MB")

    @pytest.mark.slow
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("precision_mode", PRECISION_MODES.keys())
    def test_backward_peak_memory(self, seq_len: int, precision_mode: str) -> None:
        device = jax.local_devices()[0]
        tokens = create_benchmark_tokens(BATCH_SIZE, seq_len)
        transformer, time_input = _make_transformer(tokens, precision_mode)

        def loss_fn(model):
            output = model(tokens=tokens, time=time_input, deterministic=True)
            return jnp.mean(output ** 2)

        grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))

        # Warmup / compile
        for _ in range(N_WARMUP):
            loss, _ = grad_fn(transformer)
            loss.block_until_ready()

        stats = device.memory_stats()
        if stats is None:
            pytest.skip("memory_stats() unavailable (platform allocator?)")

        peak_mb = stats["peak_bytes_in_use"] / (1024 * 1024)
        print(f"\nBackward peak memory (seq_len={seq_len}, {precision_mode}): {peak_mb:.2f} MB")
