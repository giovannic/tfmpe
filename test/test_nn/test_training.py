import diffrax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import _cfm_loss
from tfmpe.nn.training import fit_fast, fit_memory_efficient
from tfmpe.preprocessing.tokens import Tokens
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

@pytest.fixture
def training_data() -> Tokens:
    """Hierarchical Gaussian training data (params, context)."""
    rng = jax.random.PRNGKey(42)
    return create_hierarchical_gaussian_data(
        rng=rng,
        n_groups=10,
        n_obs=20,
        n_samples=10,
    )

@pytest.fixture
def validation_data() -> Tokens:
    """Hierarchical Gaussian validation data (params, context)."""
    rng = jax.random.PRNGKey(100)
    return create_hierarchical_gaussian_data(
        rng=rng,
        n_groups=10,
        n_obs=20,
        n_samples=3,
    )

@pytest.fixture
def tfmpe_instance(training_data: Tokens) -> TFMPE:
    """TFMPE with Transformer vector field."""
    config = TransformerConfig(
        latent_dim=32,
        n_encoder=1,
        n_heads=1,
        n_ff=1,
        label_dim=2
    )

    rngs = nnx.Rngs(
        params=jax.random.PRNGKey(0),
        dropout=jax.random.PRNGKey(1),
    )
    transformer = Transformer(
        config=config,
        tokens=training_data,
        rngs=rngs,
    )

    base_dist = NormalDistribution(rngs=rngs)

    return TFMPE(
        vf_network=transformer,
        base_dist=base_dist,
        solver=diffrax.Dopri5(),
        ode_kwargs={'rtol': 1e-3, 'atol': 1e-3},
    )


class TestE2ETrainingFast:
    """Test fit_fast() speed-optimized training on E2E problem."""

    @pytest.mark.slow
    def test_fit_fast_trains_successfully(
        self, tfmpe_instance: TFMPE, training_data: tuple,
        validation_data: tuple
    ) -> None:
        """Test fit_fast() trains without errors.

        Given:
        - TFMPE instance
        - Training data (10 samples)
        - Validation data (3 samples)

        When:
        - Call fit_fast() for 5 iterations with batch_size=5

        Then:
        - Training completes without errors
        - Returned losses have shape (5, 2)
        - Both train and val losses are finite
        """
        optimizer = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(tfmpe_instance, optimizer, wrt=nnx.Param)
        rng = jax.random.PRNGKey(42)

        trained_tfmpe, losses = fit_fast(
            model=tfmpe_instance,
            train=training_data,
            val=validation_data,
            loss=_cfm_loss,
            opt=opt,
            n_iter=5,
            batch_size=5,
            rng=rng,
        )

        # Check losses shape (tuple of train and val losses)
        assert isinstance(losses, tuple), (
            f"Expected losses to be tuple, "
            f"got {type(losses)}"
        )
        assert len(losses) == 2, (
            f"Expected 2 loss arrays, got {len(losses)}"
        )
        train_losses, val_losses = losses
        assert train_losses.shape == (5,), (
            f"Expected train_losses shape (5,), "
            f"got {train_losses.shape}"
        )
        assert val_losses.shape == (5,), (
            f"Expected val_losses shape (5,), "
            f"got {val_losses.shape}"
        )

        # Check losses are finite
        assert jnp.all(jnp.isfinite(train_losses)), (
            "Train losses contain NaN or Inf"
        )
        assert jnp.all(jnp.isfinite(val_losses)), (
            "Val losses contain NaN or Inf"
        )

        # Check TFMPE instance returned
        assert isinstance(trained_tfmpe, TFMPE)

    @pytest.mark.slow
    def test_fit_fast_attains_a_suitably_low_loss(
        self,
        tfmpe_instance: TFMPE,
        training_data: Tokens,
        validation_data: Tokens
    ) -> None:
        """Test fit_fast() trains without errors.

        Given:
        - TFMPE instance
        - Training data (10 samples)
        - Validation data (3 samples)

        When:
        - Call fit_fast() for 5 iterations with batch_size=5

        Then:
        - Training completes without errors
        - Returned losses have shape (5, 2)
        - Both train and val losses are finite
        """
        optimizer = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(tfmpe_instance, optimizer, wrt=nnx.Param)
        rng = jax.random.PRNGKey(42)
        threshold = 1e-1

        _, losses = fit_fast(
            model=tfmpe_instance,
            train=training_data,
            val=validation_data,
            loss=_cfm_loss,
            opt=opt,
            n_iter=1000,
            batch_size=5,
            rng=rng,
        )

        train_losses, val_losses = losses
        assert train_losses[-1] < threshold
        assert val_losses[-1] < threshold

class TestE2ETrainingMemoryEfficient:
    """Test fit_memory_efficient() memory-efficient training."""

    @pytest.mark.slow
    def test_fit_memory_efficient_trains_successfully(
        self, tfmpe_instance, training_data, validation_data
    ):
        """Test fit_memory_efficient() trains without errors.

        Given:
        - TFMPE instance
        - TokenGenerator for batches
        - Validation data

        When:
        - Call fit_memory_efficient() for 5 iterations

        Then:
        - Training completes without errors
        - Returned losses have shape (5, 2) or fewer
        - Both train and val losses are finite
        """
        optimizer = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(tfmpe_instance, optimizer, wrt=nnx.Param)
        rng = jax.random.PRNGKey(42)

        trained_tfmpe, losses = fit_memory_efficient(
            model=tfmpe_instance,
            train=training_data,
            val=validation_data,
            opt=opt,
            loss=_cfm_loss,
            n_iter=5,
            rng=rng,
            batch_size=1,
            patience=10,
        )

        # Check losses shape
        assert len(losses) == 2
        assert losses[0].shape[0] <= 5

        # Check losses are finite
        assert jnp.all(jnp.isfinite(losses[0])), (
            "Losses contain NaN or Inf"
        )

        assert isinstance(trained_tfmpe, TFMPE)
