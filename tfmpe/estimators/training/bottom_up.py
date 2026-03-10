"""Bottom-up multi-round training for TFMPE models."""

from typing import Callable, Dict, List, Tuple, Optional
from math import prod

import jax
from jax import tree, numpy as jnp
from flax import nnx
from jaxtyping import Array, PRNGKeyArray, PyTree

from ..tfmpe import TFMPE
from ..proposals import truncated_proposal_sir
from ...preprocessing.tokens import Tokens
from ...preprocessing.combine import combine_tokens
from ...preprocessing.utils import Labeller
from ...nn.training import fit_memory_efficient

from . import _cfm_loss


def fit_bottom_up(
    tfmpe: TFMPE | List[TFMPE],
    y_obs: Dict[str, Array],
    simulator_fn: Callable,
    prior_fn: Callable,
    local_fn: Callable,
    global_names: List[str],
    n_groups: int,
    n_rounds: int,
    n_samples_per_round: int,
    n_val_samples: int,
    opt: nnx.Optimizer | List[nnx.Optimizer],
    n_iter_per_round: int,
    batch_size: int,
    rng: PRNGKeyArray,
    labeller: Labeller,
    prior_log_prob: Callable[[PyTree], float],
    prob_transform: Optional[Callable] = None,
    obs_f_in: Optional[Dict] = None,
    f_in_fn: Optional[Callable]=None,
    f_in_args: list=[],
    f_in_args_global: list=[],
    epsilon: float = 1e-3
) -> Tuple[TFMPE, List[Tuple[Array, Array, Array, Array]]]:
    """Multi-round bottom-up training algorithm.

    Each round alternates between local likelihood training
    (n=1 local groups) and global posterior training
    (n=n_groups local groups).

    Currently only supports n_rounds=1. Each round makes two
    fit_memory_efficient() calls:
    1. Train p(y|theta) with n=1 local parameters
    2. Train p(theta,z|y) with n=n_groups local parameters

    This is a non-jittable wrapper using fit_memory_efficient() internally
    for each training step.

    Parameters
    ----------
    tfmpe : TFMPE
        TFMPE model to train
    y_obs : Dict[str, Array]
        Observed data with keys matching simulator output
    simulator_fn : Callable
        Function: (rng, params_dict, n) -> observations_dict
    prior_fn : Callable
        Function: (rng, n, n_samples) -> parameters_dict
    local_fn : Callable
        Function: (rng, global_samples, n) -> local_params_dict
    global_names : List[str]
        Names of global parameters (non-local)
    n_groups : int
        Number of local groups in full hierarchical model
    n_rounds : int
        Number of training rounds (currently only 1 supported)
    n_samples_per_round : int
        Number of parameter samples per round
    n_val_samples : int
        Number of validation samples
    opt : nnx.Optimizer
        NNX optimizer instance (pre-initialized with tfmpe)
    n_iter_per_round : int
        Training iterations per round
    batch_size : int
        Number of samples per batch for fit_memory_efficient calls
    rng : PRNGKeyArray
        PRNG key for sampling
    independence : Independence
        Independence structure for token creation
    labeller : Labeller
        Labeller instance with label mapping for all parameter and
        observation keys. Must include all possible keys from prior_fn,
        simulator_fn, and local_fn outputs.

    Returns
    -------
    Tuple[TFMPE, List[Tuple[Array, Array, Array, Array]]]
        Trained TFMPE and list of 4-tuples (train_loss_local,
        val_loss_local, train_loss_global, val_loss_global),
        one per round, where each loss array has shape
        (n_iter_per_round,)

    Raises
    ------
    ValueError
        If n_rounds < 1
    NotImplementedError
        If n_rounds > 1

    Notes
    -----
    - Not jittable: uses Python loop over rounds
    - Currently only n_rounds=1 is implemented
    - Each round makes TWO fit_memory_efficient() calls
    - Parameter progression: n=1 local -> n=n_groups local
    - Return value: 4-tuple of losses per round (local & global)
    """
    # Validate inputs
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")

    all_losses = []

    rng, key_prior = jax.random.split(rng)
    rng, key_sim = jax.random.split(rng)

    r = 0
    like_train_tokens = None

    while r < n_rounds:
        if isinstance(tfmpe, list) and isinstance(opt, list):
            tfmpe_local, tfmpe_global = tfmpe
            local_opt, global_opt = opt
        elif isinstance(tfmpe, TFMPE) and isinstance(opt, nnx.Optimizer):
            tfmpe_local, tfmpe_global = tfmpe, tfmpe
            local_opt, global_opt = opt, opt
        else:
            raise ValueError("models and opt need to match")

        # Compute proposal
        if r == 0:
            # Sample theta_local from prior with n=1
            if f_in_fn is not None:
                rng, key_f_in = jax.random.split(rng)
                f_in = f_in_fn(key_f_in, n_samples_per_round, *f_in_args)
            else:
                f_in = None

            theta = prior_fn(
                key_prior,
                1,
                n_samples_per_round,
                f_in
            )

            # Simulate observations
            y = simulator_fn(key_sim, theta, 1, f_in)
        else:
            rng, key_prop = jax.random.split(rng)
            theta = truncated_proposal_sir(
                key_prop,
                tfmpe_global,
                labeller,
                obs_f_in,
                n_samples_per_round,
                epsilon,
                y_obs,
                prior_fn,
                n_groups,
                prior_log_prob,
                prob_transform,
                n_batch=1_000,
                n_estimate=1_000,
            )
            theta_local = tree.map(
                lambda leaf: leaf[:, :1],
                {k: v for k, v in theta.items() if k not in global_names}
            )
            theta_global = {
                k: v
                for k, v in theta.items()
                if k in global_names
            }
            theta = {**theta_global, **theta_local}
            f_in = obs_f_in
            y = simulator_fn(key_sim, theta, 1, obs_f_in)

        # Learn local likelihood
        if like_train_tokens is None:
            like_train_tokens = Tokens.from_pytree(
                {**y, **theta},
                condition=list(theta.keys()),
                labeller=labeller,
                sample_ndims=1,
                functional_inputs=f_in
            )
        else:
            new_like_train_tokens = Tokens.from_pytree(
                {**y, **theta},
                condition=list(theta.keys()),
                labeller=labeller,
                sample_ndims=1,
                functional_inputs=f_in
            )
            like_train_tokens = combine_tokens(
                new_like_train_tokens,
                like_train_tokens
            )

        # Create validation tokens
        if f_in_fn is not None:
            rng, key_f_in = jax.random.split(rng)
            val_f_in = f_in_fn(key_f_in, n_val_samples, *f_in_args)
        else:
            val_f_in = None

        rng, key_val_prior = jax.random.split(rng)
        rng, key_val_sim = jax.random.split(rng)
        val_theta = prior_fn(
            key_val_prior,
            1,
            n_val_samples,
            val_f_in
        )
        val_y = simulator_fn(
            key_val_sim,
            val_theta,
            1,
            val_f_in
        )

        val_tokens = Tokens.from_pytree(
            {**val_y, **val_theta},
            condition=list(val_theta.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=val_f_in
        )
        print('fitting to theta')

        # First fit_fast: p(y|theta) with n=1
        tfmpe_local.train()
        rng, key_fit = jax.random.split(rng)
        tfmpe_local, first_losses = fit_memory_efficient(
            model=tfmpe_local,
            train=like_train_tokens,
            val=val_tokens,
            opt=local_opt,
            loss=_cfm_loss,
            n_iter=n_iter_per_round,
            batch_size=batch_size,
            rng=key_fit,
            patience=100,
            delta=1e-3
        )
        train_loss_local, val_loss_local = first_losses
        print("train_loss_local", train_loss_local)
        print("val_loss_local", val_loss_local)

        # Extract globals and expand to n=n_groups
        if f_in_fn is not None:
            rng, key_local_f_in, key_val_f_in = jax.random.split(rng, 3)
            f_in = f_in_fn(
                key_local_f_in,
                n_samples_per_round,
                *f_in_args_global
            )
            val_f_in = f_in_fn(
                key_val_f_in,
                n_val_samples,
                *f_in_args_global
            )
            f_in_local = {
                k: v.reshape(
                    (prod(v.shape[:2]), 1) + v.shape[2:]
                )
                for k, v in f_in.items()
                if k not in global_names
            }
            f_in_global = {
                k: jnp.repeat(v, n_groups, 0)
                for k, v in f_in.items()
                if k in global_names
            }
            f_in_reshaped = {
                **f_in_global,
                **f_in_local
            }
        else:
            f_in_reshaped = None
            val_f_in = None

        theta_global = {k: v for k, v in theta.items() if k in global_names}
        rng, key_local = jax.random.split(rng)
        theta_local = local_fn(
            key_local,
            theta_global,
            n_groups,
            f_in
        )
        single_theta_local = tree.map(
            lambda leaf: leaf.reshape(
                (prod(leaf.shape[:2]), 1) + leaf.shape[2:]
            ), # (n_samples, n_groups, n_events, n_batch) -> (n_samples * n_groups, 1, n_events, n_batch)
            theta_local
        )
        single_theta_global = tree.map(
            lambda leaf: jnp.repeat(leaf, n_groups, 0), # (n_samples, n_events, n_batch) -> (n_samples * n_groups, n_events, n_batch)
            theta_global
        )

        single_theta_n = {**single_theta_global, **single_theta_local}

        # Create param template for sampling with n=n_groups structure
        y_template = tree.map(
            lambda leaf: jnp.zeros(
                (leaf.shape[0] * n_groups, 1) + leaf.shape[2:]
            ),
            y
        )

        tokens, decoder = Tokens.from_pytree_with_decoder(
            {**y_template, **single_theta_n},
            condition=list(single_theta_n.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=f_in_reshaped,
        )

        print('sampling y_n')

        y_n = tfmpe_local.sample_posterior_batched(
            tokens,
            batch_size=100
        )

        # Create training tokens for second fit
        theta_n = {**theta_global, **theta_local}

        decoded_y_n = decoder(y_n)
        decoded_y_n = {k: v for k, v in decoded_y_n.items() if k in y.keys()}
        y_n_reshaped = tree.map(
            lambda leaf: leaf.reshape(
                (leaf.shape[0] // n_groups, n_groups) + leaf.shape[2:]
            ),
            decoded_y_n
        )

        global_train_tokens = Tokens.from_pytree(
            {**theta_n, **y_n_reshaped},
            condition=list(y_n_reshaped.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=f_in
        )

        if isinstance(tfmpe, TFMPE):
            global_train_tokens = combine_tokens(
                like_train_tokens,
                global_train_tokens
            )

        # Train global posterior
        # Create validation tokens for second fit
        rng, key_val_prior = jax.random.split(rng)
        rng, key_val_sim = jax.random.split(rng)
        val_theta = prior_fn(
            key_val_prior,
            n_groups,
            n_val_samples,
            val_f_in
        )
        val_y = simulator_fn(
            key_val_sim,
            val_theta,
            n_groups,
            val_f_in
        )

        val_tokens = Tokens.from_pytree(
            {**val_theta, **val_y},
            condition=list(val_y.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=val_f_in
        )

        # Second fit_fast (back to training mode)
        tfmpe_global.train()
        rng, key_fit = jax.random.split(rng)
        print('fit_memory_efficient')
        tfmpe_global, second_losses = fit_memory_efficient(
            model=tfmpe_global,
            train=global_train_tokens,
            val=val_tokens,
            loss=_cfm_loss,
            opt=global_opt,
            n_iter=n_iter_per_round,
            batch_size=batch_size,
            rng=key_fit,
            patience=100,
            delta=1e-3
        )
        train_loss_global, val_loss_global = second_losses

        # Append 4-tuple of losses
        all_losses.append((
            train_loss_local,
            val_loss_local,
            train_loss_global,
            val_loss_global,
        ))

        r += 1

    if isinstance(tfmpe, list):
        return tfmpe[1], all_losses

    return tfmpe, all_losses
