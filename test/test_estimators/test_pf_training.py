"""Smoke test for fit_pf posterior factorised training."""

import diffrax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_pf
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Labeller
from tfmpe.nn.transformer import Transformer, TransformerConfig


class TestFitPF:
    n_groups: int = 2
    n_obs: int = 2

    @pytest.fixture
    def prior_fn(self):
        def _prior_fn(rng, n, n_samples=10, *args):
            k1, k2 = jax.random.split(rng)
            sigma = jnp.abs(jax.random.normal(k1, (n_samples,))) + 0.1
            sigma = sigma[:, None, None]
            mu = jax.random.normal(k2, (n_samples, n))
            mu = mu[..., None]
            return {'sigma': sigma, 'mu': mu}
        return _prior_fn

    @pytest.fixture
    def simulator_fn(self):
        def _simulator_fn(rng, params_dict, n, *args):
            sigma = params_dict['sigma']
            mu = params_dict['mu']
            n_samples = sigma.shape[0]
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (n_samples, n, self.n_obs))
            y = noise * sigma + mu
            y = y[..., None]
            return {'y': y}
        return _simulator_fn

    @pytest.fixture
    def local_fn(self):
        def _local_fn(rng, global_samples, n, *args):
            n_samples = global_samples['sigma'].shape[0]
            mu = jax.random.normal(rng, (n_samples, n, 1))
            return {'mu': mu}
        return _local_fn

    @pytest.fixture
    def labeller(self):
        return Labeller.for_keys(['sigma', 'mu', 'y'])

    def _make_tfmpe(self, prior_fn, simulator_fn, labeller, n_groups):
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        params = prior_fn(key, n=n_groups, n_samples=10)
        rng, key = jax.random.split(rng)
        y = simulator_fn(key, params, n=n_groups)

        tokens = Tokens.from_pytree(
            {**params, **y},
            condition=list(y.keys()),
            sample_ndims=1,
            labeller=labeller,
        )

        config = TransformerConfig(
            latent_dim=16, n_encoder=1, n_heads=1, n_ff=2,
            attention='softmax',
        )
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))
        transformer = Transformer(config=config, tokens=tokens, rngs=rngs)
        base_dist = NormalDistribution(rngs=rngs)
        return TFMPE(vf_network=transformer, base_dist=base_dist, solver=diffrax.Dopri5())

    @pytest.mark.slow
    def test_fit_pf_runs(self, prior_fn, simulator_fn, local_fn, labeller):
        """fit_pf completes without errors and returns expected shapes."""
        # Global estimator sees y -> theta_g (sigma)
        # Need to build with varying n_groups tokens shape (use max)
        tfmpe_global = self._make_tfmpe(prior_fn, simulator_fn, labeller, self.n_groups)

        # Local estimator sees (y_single, theta_g) -> theta_l (mu)
        # Build with n=1 shape
        tfmpe_local = self._make_tfmpe(prior_fn, simulator_fn, labeller, 1)

        opt_global = nnx.Optimizer(tfmpe_global, optax.adam(1e-3), wrt=nnx.Param)
        opt_local = nnx.Optimizer(tfmpe_local, optax.adam(1e-3), wrt=nnx.Param)

        rng = jax.random.PRNGKey(42)

        trained_global, trained_local, losses = fit_pf(
            tfmpe_global=tfmpe_global,
            tfmpe_local=tfmpe_local,
            simulator_fn=simulator_fn,
            prior_fn=prior_fn,
            local_fn=local_fn,
            global_names=['sigma'],
            n_groups=self.n_groups,
            n_samples=100,
            n_val_samples=20,
            opt_global=opt_global,
            opt_local=opt_local,
            n_iter=5,
            batch_size=20,
            rng=rng,
            labeller=labeller,
        )

        assert isinstance(trained_global, TFMPE)
        assert isinstance(trained_local, TFMPE)

        (global_train, global_val), (local_train, local_val) = losses
        assert jnp.all(jnp.isfinite(global_train))
        assert jnp.all(jnp.isfinite(global_val))
        assert jnp.all(jnp.isfinite(local_train))
        assert jnp.all(jnp.isfinite(local_val))
