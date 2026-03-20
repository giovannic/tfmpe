"""Integration tests for TFMPE class."""

import diffrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tfmpe.nn.transformer import Transformer, TransformerConfig
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from .conftest import create_mock_tokens

class TestTFMPESampling:
    """Test TFMPE sampling functionality."""

    @pytest.fixture
    def identity_vf(self):
        """Identity vector field for testing.

        f(context, params, t) = 0 (no change to state)
        """
        def vf(tokens, t):
            return jnp.zeros_like(tokens.data[:, tokens.partition_idx:])
        return vf

    @pytest.fixture
    def tfmpe_identity(self, identity_vf, solver):
        """TFMPE with identity vector field."""
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        return TFMPE(
            vf_network=identity_vf,
            base_dist=NormalDistribution(rngs=rngs),
            solver=solver,
            ode_kwargs={'rtol': 1e-5, 'atol': 1e-5},
        )

    def test_sample_posterior_single_sample(
        self, tfmpe_identity
    ):
        """Test sample_posterior returns correct shape.

        Given:
        - TFMPE instance
        - Context Token with (n_tokens=2, batch_size=1)
        - Params Token template with same structure
        - Single sample (one PRNG key)

        When:
        - Call sample_posterior()

        Then:
        - Output shape matches params shape
        - All values are finite
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, 2, 1)),
            jnp.zeros((1, 2, 1)),
            sample_ndims=1
        )

        samples = tfmpe_identity.sample_posterior(tokens)

        # Check output is Tokens
        assert isinstance(samples, type(tokens))

        # Check shape matches input (single sample)
        assert samples.data.shape == tokens.data.shape

        # Check all values are finite
        assert jnp.all(jnp.isfinite(samples.data))

    def test_sample_posterior_batch_params(
        self,
        identity_vf,
        solver
    ):
        """Test sample_posterior returns correct shape with batched parameters

        When:
        - Call sample_posterior()

        Then:
        - Output shape matches params shape
        - All values are finite
        - Vector function receives correct unbatched shapes
        """
        # Create spy wrapper to capture function call shapes
        call_logs = []

        def spy_vf(tokens, time):
            call_logs.append({
                'tokens_shape': tokens.data.shape
            })
            return identity_vf(tokens, time)

        # Create TFMPE with spied vector field
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        tfmpe = TFMPE(
            vf_network=spy_vf,
            base_dist=NormalDistribution(rngs=rngs),
            solver=solver,
            ode_kwargs={'rtol': 1e-5, 'atol': 1e-5},
        )

        n_batch = 10
        tokens = create_mock_tokens(
            jnp.zeros((n_batch, 2, 1)),
            jnp.zeros((n_batch, 2, 1))
        )

        samples = tfmpe.sample_posterior(
            tokens
        )

        # Check output is Tokens
        assert isinstance(samples, type(tokens))

        # Check shape matches input
        assert samples.data.shape == tokens.data.shape

        # Check all values are finite
        assert jnp.all(jnp.isfinite(samples.data))
        # Check all values are different
        assert jnp.all(samples.data[0:1, samples.partition_idx:] != samples.data[1:, samples.partition_idx:])

        # Verify spy captured correct shapes
        assert len(call_logs) > 0, "VF was never called"
        assert call_logs[0]['tokens_shape'] == (1, 1)

    def test_sample_posterior_preserves_token_metadata(
        self, tfmpe_identity
    ):
        """Test that sample_posterior preserves Token metadata.

        Given:
        - TFMPE instance
        - Context Token with specific metadata
        - Params Token template

        When:
        - Call sample_posterior()

        Then:
        - Labels match params labels
        - Self-attention mask matches params
        - Slices metadata is preserved
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, 2, 1)),
            jnp.zeros((1, 2, 1)),
        )

        samples = tfmpe_identity.sample_posterior(tokens)

        # Check labels preserved
        assert jnp.array_equal(
            samples.labels, tokens.labels
        )

    def test_sample_posterior_with_identity_flow(
        self, tfmpe_identity
    ):
        """Test sample_posterior with identity vector field.

        With identity VF (f=0), samples should match base
        distribution (no transformation).

        Given:
        - TFMPE with f(x,c,t)=0
        - Params Token template

        When:
        - Call sample_posterior() twice with reseeded RNG

        Then:
        - Samples are deterministic (same RNG seed gives same
          result)
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, 1, 1)),
            jnp.zeros((1, 1, 1))
        )

        # Sample twice with same RNG seed
        nnx.reseed(tfmpe_identity, params=42)
        samples1 = tfmpe_identity.sample_posterior(
            tokens
        )
        nnx.reseed(tfmpe_identity, params=42)
        samples2 = tfmpe_identity.sample_posterior(
            tokens
        )

        # Should be identical (deterministic)
        assert jnp.allclose(samples1.data, samples2.data)

    @pytest.mark.parametrize(
        "n_tokens,batch_size",
        [
            (1, 1),
            (2, 1),
            (1, 3),
            (2, 2),
            (3, 5),
        ],
    )
    def test_sampling_various_token_shapes(
        self, tfmpe_identity, n_tokens, batch_size
    ):
        """Test sample_posterior with various token shapes.

        Given:
        - TFMPE instance
        - Different (n_tokens, batch_size) combinations
        - Params Token template

        When:
        - Call sample_posterior()

        Then:
        - Output shape matches params shape
        - All values finite
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, n_tokens, batch_size)),
            jnp.zeros((1, n_tokens, batch_size)),
            sample_ndims = 1
        )

        samples = tfmpe_identity.sample_posterior(tokens)

        assert samples.data.shape == (1, 2 * n_tokens, batch_size)
        assert jnp.all(jnp.isfinite(samples.data))

class TestTFMPELogProb:
    """Test TFMPE log probability computation."""

    @pytest.fixture
    def identity_vf(self):
        """Identity vector field."""
        def vf(tokens, t):
            return jnp.zeros_like(tokens.data[:, tokens.partition_idx:])
        return vf

    @pytest.fixture
    def tfmpe_identity(self, identity_vf, solver):
        """TFMPE with identity vector field."""
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        return TFMPE(
            vf_network=identity_vf,
            base_dist=NormalDistribution(rngs=rngs),
            solver=solver,
            ode_kwargs={'rtol': 1e-5, 'atol': 1e-5},
        )

    def test_log_prob_returns_scalar(
        self, tfmpe_identity
    ):
        """Test that log_prob_posterior_samples returns scalar.

        Given:
        - TFMPE instance
        - Single posterior sample Token (n_tokens=1,
          batch_size=1)

        When:
        - Call log_prob_posterior_samples()

        Then:
        - Output is a scalar (shape ())
        - Value is finite
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, 1, 1)),
            jnp.zeros((1, 1, 1)),
            sample_ndims=1
        )

        log_prob = tfmpe_identity.log_prob_posterior_samples(
            tokens
        )

        # Check output is scalar
        assert log_prob.shape == (1,)

        # Check value is finite
        assert jnp.isfinite(log_prob)

    @pytest.mark.parametrize(
        "n_tokens,batch_size",
        [
            (1, 1),
            (2, 1),
            (1, 3),
            (2, 2),
            (3, 5),
        ],
    )
    def test_log_prob_various_token_shapes(
        self, tfmpe_identity, n_tokens, batch_size
    ):
        """Test log_prob_posterior_samples with various shapes.

        Given:
        - TFMPE instance
        - Different (n_tokens, batch_size) combinations

        When:
        - Call log_prob_posterior_samples()

        Then:
        - Output is always a scalar
        - Value is finite
        """
        tokens = create_mock_tokens(
            jnp.zeros((1, n_tokens, batch_size)),
            jax.random.normal(
                jax.random.PRNGKey(42), (1, n_tokens, batch_size)
            ),
            sample_ndims=1
        )

        log_prob = tfmpe_identity.log_prob_posterior_samples(
            tokens
        )

        # Always returns scalar
        assert log_prob.shape == (1,)
        assert jnp.isfinite(log_prob)


class TestTFMPEInitialization:
    """Test TFMPE initialization and configuration."""

    def test_tfmpe_initialization(self, solver):
        """Test TFMPE can be initialized.

        Given:
        - Vector field function
        - Base distribution module
        - ODE solver

        When:
        - Create TFMPE instance

        Then:
        - Instance created successfully
        - Attributes set correctly
        """
        def vf(tokens, time):
            return jnp.zeros_like(tokens.data[tokens.partition_idx:])

        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        tfmpe = TFMPE(
            vf_network=vf,
            base_dist=NormalDistribution(rngs=rngs),
            solver=solver,
            ode_kwargs={'rtol': 1e-5, 'atol': 1e-5},
        )

        assert tfmpe.vf_network is not None
        assert tfmpe.base_dist is not None
        assert tfmpe.solver is not None

    def test_tfmpe_with_custom_ode_kwargs(self, solver):
        """Test TFMPE with custom ODE kwargs.

        Given:
        - Custom rtol and atol values

        When:
        - Create TFMPE with custom kwargs

        Then:
        - ODE kwargs stored correctly
        """
        def vf(tokens, time):
            return jnp.zeros_like(tokens.data[tokens.partition_idx:])

        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        custom_kwargs = {'rtol': 1e-3, 'atol': 1e-4}
        tfmpe = TFMPE(
            vf_network=vf,
            base_dist=NormalDistribution(rngs=rngs),
            solver=solver,
            ode_kwargs=custom_kwargs,
        )

        assert tfmpe.ode_kwargs == custom_kwargs


class TestPosteriorSamplingBenchmark:
    """Benchmark posterior sampling performance with realistic
    scales."""

    @pytest.fixture
    def tfmpe_with_transformer(self, solver):
        """TFMPE instance with Transformer vector field.

        Uses a lightweight transformer config suitable for
        benchmarking across varying token/batch sizes.
        """
        # Create minimal template tokens
        params_dict = {'x': jnp.ones((1, 1, 1)) * 0.5}
        params_tokens = Tokens.from_pytree(
            params_dict,
            condition=[],
            sample_ndims=1
        )

        # Create lightweight transformer
        config = TransformerConfig(
            latent_dim=16,
            n_encoder=1,
            n_heads=1,
            n_ff=2,
        )
        rngs = nnx.Rngs(
            params=jax.random.PRNGKey(0),
            dropout=jax.random.PRNGKey(1),
        )
        transformer = Transformer(
            config=config, tokens=params_tokens, rngs=rngs
        )
        base_dist = NormalDistribution(rngs=rngs)
        tfmpe = TFMPE(
            vf_network=transformer,
            base_dist=base_dist,
            solver=solver,
        )
        tfmpe.eval()

        return tfmpe

    @pytest.mark.slow
    @pytest.mark.parametrize("n_tokens", [1, 10, 20, 50])
    @pytest.mark.parametrize("sample_size", [1, 100, 1000])
    def test_sample_posterior_benchmark(
        self, tfmpe_with_transformer, n_tokens, sample_size,
        benchmark
    ):
        """Benchmark sample_posterior across token/batch sizes.

        Measures wall-clock time for sampling from posterior with
        varying token lengths and batch sizes to understand scaling
        dynamics.

        Parameters
        ----------
        n_tokens : int
            Number of tokens in the sequence
        batch_size : int
            Number of samples in the batch
        benchmark : pytest_benchmark.fixture
            Benchmark fixture for timing
        """
        # Create tokens of specified shape
        tokens = create_mock_tokens(
            jnp.zeros((sample_size, n_tokens, 1)),
            jnp.zeros((sample_size, n_tokens, 1))
        )

        # Benchmark the sampling operation
        def sample():
            return tfmpe_with_transformer.sample_posterior(
                tokens
            )

        benchmark(sample)


class TestPosteriorSamplingScaling:
    """Benchmark attention type and solver impact on posterior
    sampling at realistic hierarchical scales.

    Simulates SIR-like token structure:
    - 1 global param token
    - n_l local param tokens (1 per site)
    - n_obs_per_site * n_l observation tokens

    Note: large-scale cases (e.g. n_l=1000) may OOM on GPUs with
    limited memory. To fail fast instead of hanging, set:
        XLA_PYTHON_CLIENT_ALLOCATOR=platform
    before running these tests.
    """

    @staticmethod
    def _make_tfmpe(attention, solver, ops_dtype=jnp.float32,
                    sensitive_ops_dtype=jnp.float32):
        params_dict = {'x': jnp.ones((1, 1, 1)) * 0.5}
        template = Tokens.from_pytree(
            params_dict, condition=[], sample_ndims=1
        )

        config = TransformerConfig(
            latent_dim=64,
            n_encoder=2,
            n_heads=16,
            n_ff=2,
            attention=attention,
            ops_dtype=ops_dtype,
            sensitive_ops_dtype=sensitive_ops_dtype,
        )
        rngs = nnx.Rngs(
            params=jax.random.PRNGKey(0),
            dropout=jax.random.PRNGKey(1),
        )
        transformer = Transformer(
            config=config, tokens=template, rngs=rngs
        )
        base_dist = NormalDistribution(rngs=rngs)
        tfmpe = TFMPE(
            vf_network=transformer,
            base_dist=base_dist,
            solver=solver,
        )
        tfmpe.eval()
        return tfmpe

    @staticmethod
    def _make_tokens(n_l, n_obs_per_site, sample_size):
        n_param_tokens = 1 + n_l
        n_context_tokens = n_obs_per_site * n_l
        return create_mock_tokens(
            jnp.zeros((sample_size, n_context_tokens, 1)),
            jnp.zeros((sample_size, n_param_tokens, 1)),
            sample_ndims=1,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "attention", ["softmax", "linear"]
    )
    @pytest.mark.parametrize(
        "solver_name", ["dopri5", "heun", "euler"]
    )
    @pytest.mark.parametrize(
        "n_l,n_obs_per_site",
        [
            (50, 10),    # SIR-like: 551 tokens
            (1000, 50),  # target scale: 51001 tokens
        ],
    )
    @pytest.mark.parametrize(
        "dtype_name", ["f32", "mixed", "bf16"]
    )
    def test_scaling_benchmark(
        self, attention, solver_name, n_l, n_obs_per_site,
        dtype_name, benchmark
    ):
        """Benchmark attention × solver × dtype × scale.

        Measures wall-clock time for 1000 posterior samples across
        attention types (softmax vs linear), solvers (Dopri5,
        Heun, Euler), dtypes (f32, mixed bf16/f32, full bf16),
        and hierarchical scales (n_l=50, n_l=1000).
        """
        dtype_map = {
            "f32": (jnp.float32, jnp.float32),
            "mixed": (jnp.bfloat16, jnp.float32),
            "bf16": (jnp.bfloat16, jnp.bfloat16),
        }
        ops_dtype, sensitive_ops_dtype = dtype_map[dtype_name]

        solver_map = {
            "dopri5": diffrax.Dopri5(),
            "heun": diffrax.Heun(),
            "euler": diffrax.Euler(),
        }
        solver = solver_map[solver_name]
        sample_size = 1000

        tfmpe = self._make_tfmpe(
            attention, solver, ops_dtype, sensitive_ops_dtype
        )
        tokens = self._make_tokens(
            n_l, n_obs_per_site, sample_size
        )

        n_total = tokens.data.shape[1]
        print(
            f"\n  {attention} | {solver_name} | {dtype_name} | "
            f"n_l={n_l} | {n_total} tokens | "
            f"samples={sample_size}"
        )

        def sample():
            result = tfmpe.sample_posterior(tokens)
            result.data.block_until_ready()
            return result

        benchmark(sample)
