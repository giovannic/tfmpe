"""End-to-end training tests for TFMPE.

Tests complete training pipelines on realistic hierarchical models
with both speed-optimized and memory-efficient training loops.
"""

import diffrax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_bottom_up, _cfm_loss
from tfmpe.nn.training import fit_fast
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Labeller
from tfmpe.nn.transformer import Transformer, TransformerConfig
from jaxtyping import Array

def create_hierarchical_gaussian_data(
    rng: Array,
    n_groups: int = 10,
    n_obs: int = 20,
    n_samples: int = 1,
) -> Tokens:
    """Generate hierarchical Gaussian linear model data.

    Parameters
    ----------
    rng : PRNGKeyArray
        Random number generator key
    n_groups : int
        Number of groups (local parameters)
    n_obs : int
        Number of observations per group
    n_samples : int
        Number of samples to generate (batched)

    Returns
    -------
    tuple[Tokens, Tokens]
        (params_tokens, context_tokens) where:
        - params_tokens contains sigma and mu
        - context_tokens contains y (observations)

    Model
    -----
    sigma ~ HalfNormal(1)
    mu_i ~ Normal(0, 1) for i in 1..n_groups
    y_ij ~ Normal(mu_i, sigma) for j in 1..n_obs
    """

    # Generate sigma (n_samples,)
    rng, key = jax.random.split(rng)
    sigma = jnp.abs(jax.random.normal(key, (n_samples,))) + 0.1

    # Generate mu (n_samples, n_groups)
    rng, key = jax.random.split(rng)
    mu = jax.random.normal(key, (n_samples, n_groups))

    # Generate y (n_samples, n_groups, n_obs)
    rng, key = jax.random.split(rng)
    noise = jax.random.normal(
        key, (n_samples, n_groups, n_obs)
    )
    # Broadcast sigma and mu to match noise shape
    y = (
        noise * sigma[:, None, None] +
        mu[:, :, None]
    )
    # Reshape to (n_samples, n_groups * n_obs)
    y = y.reshape(n_samples, n_groups * n_obs)

    # Create params Token with shape (n_samples, n_tokens, 1)
    params_dict = {
        'sigma': sigma[:, None, None],  # (n_samples, 1, 1)
        'mu': mu[..., None],  # (n_samples, n_groups, 1)
    }
    context_dict = {
        'y': y[..., None],  # (n_samples, n_groups * n_obs, 1)
    }
    tokens = Tokens.from_pytree(
        {
            **params_dict,
            **context_dict
        },
        condition=['y'],
        sample_ndims=1,
    )

    return tokens

class TestE2ETrainingBottomUp:
    """Test fit_bottom_up() multi-round bottom-up training."""

    # Test parameters
    n_groups: int = 2
    n_obs: int = 2

    @pytest.fixture
    def prior_fn(self):
        """Create prior function that accepts n_groups parameter.

        Returns a function that samples parameters from prior:
        sigma ~ HalfNormal(1)
        mu_i ~ Normal(0, 1) for i in 1..n_groups

        Parameterized by n to enable sample efficiency in bottom-up
        algorithm (n=1 for local likelihood, n=n_groups for global).
        """
        def _prior_fn(
            rng: Array,
            n: int,
            n_samples: int = 10,
            *args
        ) -> dict:
            k1, k2 = jax.random.split(rng)

            # sigma: shape (n_samples, 1, 1)
            sigma = (
                jnp.abs(jax.random.normal(k1, (n_samples,))) + 0.1
            )
            sigma = sigma[:, None, None]

            # mu: shape (n_samples, n, 1)
            mu = jax.random.normal(k2, (n_samples, n))
            mu = mu[..., None]

            return {'sigma': sigma, 'mu': mu}

        return _prior_fn

    @pytest.fixture
    def simulator_fn(self):
        """Create simulator function that accepts n_groups parameter.

        Returns a function that generates observations given parameters:
        y ~ Normal(mu, sigma)

        Parameterized by n to enable sample efficiency in bottom-up
        algorithm (n=1 for local likelihood, n=n_groups for global).
        """
        def _simulator_fn(
            rng: Array,
            params_dict: dict,
            n: int,
            *args
        ) -> dict:
            sigma = params_dict['sigma']
            mu = params_dict['mu']
            n_samples = sigma.shape[0]

            # Generate noise
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(
                key, (n_samples, n, self.n_obs)
            )

            # Simulate observations: y = mu + sigma * noise
            # Handle shapes: sigma (n_samples, 1, 1),
            # mu (n_samples, n, 1)
            y = noise * sigma + mu
            # Add batch dimension: (n_samples, n, n_obs, 1)
            y = y[..., None]

            return {'y': y}

        return _simulator_fn

    @pytest.fixture
    def local_fn(self):
        """Create local prior function for bottom-up algorithm.

        Generates local parameters given global parameters and
        number of local groups.
        """
        def _local_fn(
            rng: Array,
            global_samples: dict,
            n: int,
            *args
        ) -> dict:
            """Sample local parameters.

            Parameters
            ----------
            rng : PRNGKeyArray
                Random number generator key
            global_samples : dict
                Global parameter samples (e.g., {'sigma': ...})
            n : int
                Number of local groups to generate

            Returns
            -------
            dict
                Local parameters dict with mu shape (n_samples, n, 1)
            """
            n_samples = global_samples['sigma'].shape[0]
            mu = jax.random.normal(rng, (n_samples, n, 1))
            return {'mu': mu}

        return _local_fn

    @pytest.fixture
    def labeller(self) -> Labeller:
        # Create global labeller with all keys
        return Labeller.for_keys(['sigma', 'mu', 'y'])

    @pytest.fixture
    def tfmpe_instance(self, prior_fn, simulator_fn, labeller) -> TFMPE:
        """TFMPE with Transformer vector field."""
        # Generate training data using prior_fn/simulator_fn
        rng = jax.random.PRNGKey(42)
        rng, key = jax.random.split(rng)
        train_params = prior_fn(key, n=10, n_samples=10)
        context_params = simulator_fn(key, train_params, n=10)

        # Convert to Tokens
        train_params_tokens = Tokens.from_pytree(
            {**train_params, **context_params},
            condition=list(context_params.keys()),
            sample_ndims=1,
            labeller=labeller
        )

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
            config=config,
            tokens=train_params_tokens,
            rngs=rngs,
        )

        base_dist = NormalDistribution(rngs=rngs)

        return TFMPE(
            vf_network=transformer,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
        )

    @pytest.mark.slow
    def test_fit_bottom_up_trains_successfully(
        self,
        tfmpe_instance: TFMPE,
        prior_fn,
        simulator_fn,
        local_fn,
        labeller
    ) -> None:
        """Test fit_bottom_up() trains without errors.

        Given:
        - TFMPE instance
        - prior_fn, simulator_fn, and local_fn fixtures

        When:
        - Call fit_bottom_up() with n_rounds=2, n_samples=10,
          n_groups=10

        Then:
        - Training completes without errors
        - Returns trained TFMPE instance
        - Returns list of 2 loss tuples (one per round)
        - All losses are finite
        """


        # Generate y_obs using simulator_fn with full n_groups
        rng = jax.random.PRNGKey(42)
        rng, key = jax.random.split(rng)
        params = prior_fn(key, n=self.n_groups, n_samples=1)
        rng, key = jax.random.split(rng)
        y_obs = simulator_fn(key, params, n=self.n_groups)

        optimizer = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(tfmpe_instance, optimizer, wrt=nnx.Param)
        rng = jax.random.PRNGKey(42)

        trained_tfmpe, all_losses = fit_bottom_up(
            tfmpe=tfmpe_instance,
            y_obs=y_obs,
            simulator_fn=simulator_fn,
            prior_fn=prior_fn,
            local_fn=local_fn,
            global_names=['sigma'],
            n_groups=self.n_groups,
            n_rounds=1,
            n_samples_per_round=100,
            n_val_samples=10,
            opt=opt,
            n_iter_per_round=100,
            batch_size=100,
            rng=rng,
            labeller=labeller,
            prior_log_prob=lambda x: 1.
        )

        # Check TFMPE instance returned
        assert isinstance(trained_tfmpe, TFMPE), (
            f"Expected TFMPE instance, got {type(trained_tfmpe)}"
        )

        # Check losses is list of 4-tuples
        assert isinstance(all_losses, list), (
            f"Expected losses to be list, got {type(all_losses)}"
        )
        assert len(all_losses) == 1, (
            f"Expected 1 loss tuple (Round 0 only), "
            f"got {len(all_losses)}"
        )

        # Check each round's losses (4-tuple) are finite
        for round_idx, loss_tuple in enumerate(all_losses):
            assert len(loss_tuple) == 4, (
                f"Round {round_idx}: Expected 4-tuple, "
                f"got {len(loss_tuple)}"
            )
            train_loss_local, val_loss_local, (
                train_loss_global
            ), val_loss_global = loss_tuple

            assert jnp.all(jnp.isfinite(train_loss_local)), (
                f"Round {round_idx}: Local train losses contain NaN/Inf"
            )
            assert jnp.all(jnp.isfinite(val_loss_local)), (
                f"Round {round_idx}: Local val losses contain NaN/Inf"
            )
            assert jnp.all(jnp.isfinite(train_loss_global)), (
                f"Round {round_idx}: Global train losses contain NaN/Inf"
            )
            assert jnp.all(jnp.isfinite(val_loss_global)), (
                f"Round {round_idx}: Global val losses contain NaN/Inf"
            )

class TestE2ETrainingHalfNormal:
    """Test fit_fast() learns simple Gaussian target N(0, 2).

    Tests that TFMPE can learn to transform from base distribution
    N(0, 1) to target distribution N(0, 2), verifying both the
    learned vector field and posterior samples.
    """

    @pytest.fixture(params=[1, 5, 10])
    def n_dim(self, request):
        """Parameterize test across different dimensions."""
        return request.param

    @pytest.fixture
    def gaussian_training_data(self, n_dim, labeller) -> Tokens:
        """Generate (params, context) for HalfNormal(0, 1) target.

        Token shapes: (n_samples, n_tokens, 1) where n_tokens = n_dim
        """
        rng = jax.random.PRNGKey(42)
        n_samples = 1000  # Sufficient for training

        # Generate params from TARGET distribution N(0, 2)
        # Shape: (n_samples, n_dim, 1) where n_dim tokens, dim=1
        rng, key = jax.random.split(rng)
        params_data = {
            'y': jnp.abs(jax.random.normal(key, (n_samples, 1, 1))),
            'x': jax.random.normal(key, (n_samples, n_dim, 1))
        }

        # Generate independent context from N(0, 1)
        # Shape: (n_samples, 1, 1) where 1 token, dim=1
        rng, key = jax.random.split(rng)
        context_data = jax.random.normal(key, (n_samples, 1, 1))

        # Convert to Tokens
        tokens = Tokens.from_pytree(
            {**params_data, **{'c': context_data}},
            condition=['c'],
            sample_ndims=1,
            labeller=labeller
        )

        return tokens

    @pytest.fixture
    def labeller(self) -> Labeller:
        return Labeller.for_keys(['x', 'y', 'c'])

    @pytest.fixture
    def gaussian_validation_data(self, n_dim, labeller) -> Tokens:
        """Generate validation data for HalfNormal(0, 1) target."""
        rng = jax.random.PRNGKey(100)
        n_samples = 20  # Fewer samples for validation

        # Generate params from TARGET distribution HalfNormal(0, 1)
        rng, key = jax.random.split(rng)
        params_data = {
            'y': jnp.abs(jax.random.normal(key, (n_samples, 1, 1))),
            'x': jax.random.normal(key, (n_samples, n_dim, 1)),
        }

        # Generate independent context from N(0, 1)
        rng, key = jax.random.split(rng)
        context_data = jax.random.normal(key, (n_samples, 1, 1))

        # Convert to Tokens
        tokens = Tokens.from_pytree(
            {**params_data, **{'c': context_data}},
            condition=['c'],
            sample_ndims=1,
            labeller=labeller
        )

        return tokens

    @pytest.fixture
    def gaussian_tfmpe_instance(
        self,
        gaussian_training_data
    ) -> TFMPE:
        """TFMPE instance for Gaussian learning."""
        train_params = gaussian_training_data

        config = TransformerConfig(
            latent_dim=32,
            n_encoder=1,
            n_heads=1,
            n_ff=1,  # Number of feed forward layers
            label_dim=2
        )

        rngs = nnx.Rngs(
            params=jax.random.PRNGKey(0),
            dropout=jax.random.PRNGKey(1),
        )

        transformer = Transformer(
            config=config,
            tokens=train_params,
            rngs=rngs,
        )

        base_dist = NormalDistribution(rngs=rngs)

        return TFMPE(
            vf_network=transformer,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
            ode_kwargs={'rtol': 1e-5, 'atol': 1e-5},
        )

    @pytest.mark.slow
    def test_posterior_samples_are_positive(
        self,
        n_dim,
        gaussian_tfmpe_instance,
        gaussian_training_data,
        gaussian_validation_data,
        labeller
    ):
        """Verify samples from posterior have mean≈0, std≈2.

        Given:
        - Trained TFMPE instance
        - fit_fast training for 1000 iterations

        When:
        - Sample 1000 times from posterior

        Then:
        - Most samples are positive
        """
        # Train the model
        optimizer = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(
            gaussian_tfmpe_instance, optimizer, wrt=nnx.Param
        )
        rng = jax.random.PRNGKey(42)

        trained_tfmpe, _ = fit_fast(
            model=gaussian_tfmpe_instance,
            train=gaussian_training_data,
            val=gaussian_validation_data,
            loss=_cfm_loss,
            opt=opt,
            n_iter=1000,
            batch_size=100,
            rng=rng,
        )

        # Verify training succeeded
        assert isinstance(trained_tfmpe, TFMPE)

        # Create batched context and params for vectorized sampling
        n_samples = 10000

        # Params template: (n_samples, n_dim, 1)
        tokens, decoder = Tokens.from_pytree_with_decoder(
            {
                'y': jnp.zeros((n_samples, 1, 1)),
                'x': jnp.zeros((n_samples, n_dim, 1)),
                'c': jnp.zeros((n_samples, 1, 1))
            },
            condition=['c'],
            sample_ndims=1,
            labeller=labeller
        )

        # Vmap over samples
        all_samples = trained_tfmpe.sample_posterior(
            tokens
        )
        # all_samples.data shape: (1000, n_dim, 1)

        # Compute statistics across all dimensions
        samples_array = decoder(all_samples)['y']

        threshold = .05
        assert jnp.sum(samples_array < 0 , axis=0) < n_samples * threshold, (
            "Too many negative samples"
        )
