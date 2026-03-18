"""Direct training for TFMPE models."""

from typing import Callable, Tuple, Optional

import jax
from flax import nnx
from jaxtyping import Array, PRNGKeyArray

from ..tfmpe import TFMPE
from ...preprocessing.tokens import Tokens
from ...preprocessing.utils import Labeller
from ...nn.training import fit_memory_efficient

from .loss import cfm_loss as _cfm_loss


def fit_directly(
    tfmpe: TFMPE,
    simulator_fn: Callable,
    prior_fn: Callable,
    n_groups: int,
    n_samples_per_round: int,
    n_val_samples: int,
    opt: nnx.Optimizer,
    n_iter_per_round: int,
    batch_size: int,
    rng: PRNGKeyArray,
    labeller: Labeller,
    f_in_fn: Optional[Callable] = None,
    f_in_args: list = [],
    delta: float = 0.0,
    patience: int = 0,
) -> Tuple[TFMPE, Tuple[Array, Array]]:
    """Version of fit_bottom_up which fits the global estimator directly.

    Parameters
    ----------
    tfmpe : TFMPE
        TFMPE model to train
    simulator_fn : Callable
        Function: (rng, params_dict, n, f_in) -> observations_dict
    prior_fn : Callable
        Function: (rng, n, n_samples, f_in) -> parameters_dict
    n_groups : int
        Number of local groups in full hierarchical model
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
    labeller : Labeller
        Labeller instance with label mapping for all parameter and
        observation keys.
    f_in_fn : Callable, optional
        Function to generate functional inputs: (rng, n_samples, *f_in_args) -> f_in_dict
    f_in_args : list, optional
        Additional arguments for f_in_fn
    delta : float, optional
        Minimum improvement in training loss to reset patience counter.
        Default is 0.0 (any improvement counts).
    patience : int, optional
        Number of epochs to wait for improvement before stopping.
        Set to 0 to disable early stopping. Default is 0.

    Returns
    -------
    Tuple[TFMPE, Tuple[Array, Array]]
        Trained TFMPE and tuple of (train_losses, val_losses)
    """
    rng, key_prior = jax.random.split(rng)
    rng, key_sim = jax.random.split(rng)

    # Generate functional inputs if provided
    if f_in_fn is not None:
        rng, key_f_in = jax.random.split(rng)
        f_in = f_in_fn(key_f_in, n_samples_per_round, *f_in_args)
    else:
        f_in = None

    # Sample theta from prior
    theta = prior_fn(
        key_prior,
        n_groups,
        n_samples_per_round,
        f_in
    )

    # Simulate observations
    y = simulator_fn(key_sim, theta, n_groups, f_in)

    # Create training tokens combining y and theta with condition
    train_tokens = Tokens.from_pytree(
        {**y, **theta},
        condition=list(y.keys()),
        labeller=labeller,
        sample_ndims=1,
        functional_inputs=f_in
    )

    # Create validation tokens
    if f_in_fn is not None:
        rng, key_val_f_in = jax.random.split(rng)
        val_f_in = f_in_fn(key_val_f_in, n_val_samples, *f_in_args)
    else:
        val_f_in = None

    rng, key_val_prior = jax.random.split(rng)
    rng, key_val_sim = jax.random.split(rng)
    val_theta = prior_fn(
        key_val_prior,
        n_groups,
        n_val_samples,
        val_f_in
    )
    val_y = simulator_fn(key_val_sim, val_theta, n_groups, val_f_in)

    val_tokens = Tokens.from_pytree(
        {**val_y, **val_theta},
        condition=list(val_y.keys()),
        labeller=labeller,
        sample_ndims=1,
        functional_inputs=val_f_in
    )

    # Fit p(theta|y) with n=n_groups
    tfmpe.train()
    rng, key_fit = jax.random.split(rng)
    tfmpe, losses = fit_memory_efficient(
        model=tfmpe,
        train=train_tokens,
        val=val_tokens,
        opt=opt,
        loss=_cfm_loss,
        n_iter=n_iter_per_round,
        batch_size=batch_size,
        rng=key_fit,
        delta=delta,
        patience=patience
    )
    train_loss, val_loss = losses
    print("train_loss", train_loss)
    print("val_loss", val_loss)

    return tfmpe, (train_loss, val_loss)
