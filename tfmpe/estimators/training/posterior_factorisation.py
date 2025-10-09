"""Posterior factorised training for TFMPE models.

Decomposes the hierarchical posterior as:
    p(theta_g, theta_l | y) = p(theta_g | y) * prod_s p(theta_l[s] | theta_g, y_[s])

Training proceeds sequentially:
1. Train global estimator q(theta_g | y) on prior predictive data with
   varying n_groups (1 to n), padded to max.
2. Sample the trained global estimator to get theta_g* given simulated y.
3. Train local estimator q(theta_l[s] | theta_g, y_[s]) reusing the same y
   (reshaped to single-group), with theta_g* as conditioning.
"""

from typing import Callable, List, Tuple, Optional

import jax
from jax import tree, numpy as jnp
from flax import nnx
from jaxtyping import Array, PRNGKeyArray

from ..tfmpe import TFMPE
from ...preprocessing.tokens import Tokens
from ...preprocessing.combine import combine_tokens
from ...preprocessing.utils import Labeller
from ...nn.training import fit_nn

from .loss import cfm_loss as _cfm_loss


def _sample_sizes_stick_breaking(
    rng: PRNGKeyArray, budget: int, n_groups: int
) -> Tuple[dict, PRNGKeyArray]:
    """Sample a bag of group sizes via stick-breaking (scheme S1).

    Sequentially draws n ~ Uniform{1, ..., min(n_groups, remaining)} until the
    simulation budget is exhausted. Uses the budget exactly. Returns a dict
    mapping group size -> number of rows at that size, plus the advanced rng.
    """
    counts: dict = {}
    remaining = int(budget)
    while remaining > 0:
        max_n = min(n_groups, remaining)
        rng, key = jax.random.split(rng)
        n = int(jax.random.randint(key, (), 1, max_n + 1))
        counts[n] = counts.get(n, 0) + 1
        remaining -= n
    return counts, rng


def fit_pf(
    tfmpe_global: TFMPE,
    tfmpe_local: TFMPE,
    simulator_fn: Callable,
    prior_fn: Callable,
    local_fn: Callable,
    global_names: List[str],
    n_groups: int,
    n_samples: int,
    n_val_samples: int,
    opt_global: nnx.Optimizer,
    opt_local: nnx.Optimizer,
    n_iter: int,
    batch_size: int,
    rng: PRNGKeyArray,
    labeller: Labeller,
    f_in_fn: Optional[Callable] = None,
    f_in_args: list = [],
    delta: float = 0.0,
    patience: int = 0,
    sample_batch_size: int = 1000
) -> Tuple[TFMPE, TFMPE, Tuple[Tuple[Array, Array], Tuple[Array, Array]]]:
    """Posterior factorised training with separate global and local estimators.

    Trains two TFMPE instances sequentially:
    1. Global estimator on prior predictive data with varying group counts
    2. Local estimator on single-group data conditioned on sampled globals

    Parameters
    ----------
    tfmpe_global : TFMPE
        TFMPE model for global posterior estimation
    tfmpe_local : TFMPE
        TFMPE model for local posterior estimation
    simulator_fn : Callable
        Function: (rng, params_dict, n, f_in) -> observations_dict
    prior_fn : Callable
        Function: (rng, n, n_samples, f_in) -> parameters_dict
    local_fn : Callable
        Function: (rng, global_samples, n, f_in) -> local_params_dict
    global_names : List[str]
        Names of global parameters
    n_groups : int
        Maximum number of groups; training uses uniform [1, n_groups]
    n_samples : int
        Total training samples (split evenly across group counts)
    n_val_samples : int
        Number of validation samples
    opt_global : nnx.Optimizer
        Optimizer for global estimator
    opt_local : nnx.Optimizer
        Optimizer for local estimator
    n_iter : int
        Training iterations for each estimator
    batch_size : int
        Batch size for training calls
    rng : PRNGKeyArray
        PRNG key
    labeller : Labeller
        Labeller instance for token creation
    f_in_fn : Callable, optional
        Function to generate functional inputs: (rng, n_samples, *f_in_args) -> f_in_dict
    f_in_args : list, optional
        Additional arguments for f_in_fn
    delta : float, optional
        Minimum improvement for early stopping. Default 0.0.
    patience : int, optional
        Epochs without improvement before stopping. 0 disables. Default 0.

    Returns
    -------
    Tuple[TFMPE, TFMPE, Tuple[Tuple[Array, Array], Tuple[Array, Array]]]
        (trained_global, trained_local,
         ((global_train_loss, global_val_loss),
          (local_train_loss, local_val_loss)))
    """
    # Budget: a sample at group count n costs n simulations. Sizes are drawn
    # via stick-breaking (scheme S1): n ~ Uniform{1, ..., min(n_groups,
    # remaining)} until the budget is exhausted. This uses the full budget
    # exactly and does not require a minimum of one sample per size.
    train_counts, rng = _sample_sizes_stick_breaking(rng, n_samples, n_groups)
    val_counts, rng = _sample_sizes_stick_breaking(rng, n_val_samples, n_groups)

    total_train_samples = sum(train_counts.values())
    assert total_train_samples >= batch_size, (
        f"n_samples={n_samples} too small for n_groups={n_groups}: "
        f"total_train_samples={total_train_samples} < batch_size={batch_size}."
    )

    data_per_n = {}  # n -> (theta_g, theta_l, y, f_in)
    global_train_tokens = None

    for n, count in sorted(train_counts.items()):
        rng, key_f_in, key_prior, key_sim = jax.random.split(rng, 4)

        f_in = f_in_fn(key_f_in, count, *f_in_args) if f_in_fn else None

        theta = prior_fn(key_prior, n, count, f_in)
        y = simulator_fn(key_sim, theta, n, f_in)

        theta_g = {k: v for k, v in theta.items() if k in global_names}
        theta_l = {k: v for k, v in theta.items() if k not in global_names}
        data_per_n[n] = (theta_g, theta_l, y, f_in)

        # Global tokens: condition=y, target=theta_g
        tokens_n = Tokens.from_pytree(
            {**y, **theta_g},
            condition=list(y.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=f_in,
        )
        global_train_tokens = (
            tokens_n if global_train_tokens is None
            else combine_tokens(global_train_tokens, tokens_n)
        )

    global_val_tokens = None

    for n, count in sorted(val_counts.items()):
        rng, key_f_in, key_prior, key_sim = jax.random.split(rng, 4)

        val_f_in = f_in_fn(key_f_in, count, *f_in_args) if f_in_fn else None

        val_theta = prior_fn(key_prior, n, count, val_f_in)
        val_y = simulator_fn(key_sim, val_theta, n, val_f_in)

        val_theta_g = {k: v for k, v in val_theta.items() if k in global_names}

        val_tokens_n = Tokens.from_pytree(
            {**val_y, **val_theta_g},
            condition=list(val_y.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=val_f_in,
        )
        global_val_tokens = (
            val_tokens_n if global_val_tokens is None
            else combine_tokens(global_val_tokens, val_tokens_n)
        )

    tfmpe_global.train()
    rng, key_fit_global = jax.random.split(rng)
    tfmpe_global, global_losses = fit_nn(
        model=tfmpe_global,
        train=global_train_tokens,
        val=global_val_tokens,
        opt=opt_global,
        loss=_cfm_loss,
        n_iter=n_iter,
        batch_size=batch_size,
        rng=key_fit_global,
        delta=delta,
        patience=patience,
    )

    local_train_tokens = None

    for n in sorted(data_per_n):
        theta_g, theta_l, y, f_in = data_per_n[n]

        # Create sampling tokens from {y, theta_g} with condition=y keys
        tokens_for_sampling, decoder = Tokens.from_pytree_with_decoder(
            {**y, **theta_g},
            condition=list(y.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=f_in,
        )

        # Sample global posterior
        sampled = tfmpe_global.sample_posterior_batched(
            tokens_for_sampling, batch_size=sample_batch_size
        )

        # Decode sampled tokens back to dict
        decoded = decoder(sampled)
        theta_g_star = {k: v for k, v in decoded.items() if k in global_names}

        # Reshape to single-group samples
        # theta_g*: (samples_per_n, ...) -> repeat n times -> (samples_per_n * n, ...)
        theta_g_expanded = tree.map(
            lambda v: jnp.repeat(v, n, axis=0), theta_g_star
        )

        # theta_l: (samples_per_n, n, ...) -> (samples_per_n * n, 1, ...)
        theta_l_reshaped = tree.map(
            lambda v: v.reshape((v.shape[0] * n, 1) + v.shape[2:]), theta_l
        )

        # y: (samples_per_n, n, ...) -> (samples_per_n * n, 1, ...)
        y_reshaped = tree.map(
            lambda v: v.reshape((v.shape[0] * n, 1) + v.shape[2:]), y
        )

        # Handle f_in reshaping
        local_f_in = None
        if f_in is not None:
            f_in_local = {
                k: v.reshape((v.shape[0] * n, 1) + v.shape[2:])
                for k, v in f_in.items()
                if k not in global_names
            }
            f_in_global = {
                k: jnp.repeat(v, n, axis=0)
                for k, v in f_in.items()
                if k in global_names
            }
            local_f_in = {**f_in_global, **f_in_local}

        # Create local tokens: condition = y + theta_g*, target = theta_l
        local_tokens_n = Tokens.from_pytree(
            {**y_reshaped, **theta_g_expanded, **theta_l_reshaped},
            condition=list(y_reshaped.keys()) + list(theta_g_expanded.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=local_f_in,
        )
        local_train_tokens = (
            local_tokens_n if local_train_tokens is None
            else combine_tokens(local_train_tokens, local_tokens_n)
        )

    # Local validation tokens from prior (n=1, true theta_g conditioning)
    rng, key_f_in, key_prior, key_sim = jax.random.split(rng, 4)

    val_f_in = f_in_fn(key_f_in, n_val_samples, *f_in_args) if f_in_fn else None

    val_theta = prior_fn(key_prior, 1, n_val_samples, val_f_in)
    val_y = simulator_fn(key_sim, val_theta, 1, val_f_in)

    val_theta_g = {k: v for k, v in val_theta.items() if k in global_names}
    val_theta_l = {k: v for k, v in val_theta.items() if k not in global_names}

    local_val_tokens = Tokens.from_pytree(
        {**val_y, **val_theta_g, **val_theta_l},
        condition=list(val_y.keys()) + list(val_theta_g.keys()),
        labeller=labeller,
        sample_ndims=1,
        functional_inputs=val_f_in,
    )

    tfmpe_local.train()
    rng, key_fit_local = jax.random.split(rng)
    tfmpe_local, local_losses = fit_nn(
        model=tfmpe_local,
        train=local_train_tokens,
        val=local_val_tokens,
        opt=opt_local,
        loss=_cfm_loss,
        n_iter=n_iter,
        batch_size=batch_size,
        rng=key_fit_local,
        delta=delta,
        patience=patience,
    )

    return tfmpe_global, tfmpe_local, (global_losses, local_losses)
