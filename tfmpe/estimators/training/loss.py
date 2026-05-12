"""Loss for TFMPE estimators"""

import jax
from jax import numpy as jnp
from jaxtyping import Array

from ..tfmpe import TFMPE
from ...preprocessing.tokens import Tokens

import dataclasses

def cfm_loss(
    tfmpe: TFMPE,
    tokens: Tokens,
    rng: Array,
) -> Array:
    """Continuous Flow Matching loss with batched inputs.

    Computes CFM loss for batched inputs with leading batch
    dimension. Returns scalar loss averaged over batch.

    Parameters
    ----------
    tfmpe : TFMPE
        TFMPE model instance
    tokens: Tokens
        Tokens to compute loss over
    time : Array
        Time points for batch. Shape: (batch,)

    Returns
    -------
    Array
        Scalar loss (averaged over batch)
    """
    sigma_min = 0.001
    theta_data = tokens.data[:, tokens.partition_idx:]

    # Sample from base distribution
    theta_0 = tfmpe.base_dist.sample(theta_data.shape)

    # Reshape time for broadcasting: (batch,) -> (batch, 1, 1)
    # theta_data shape: (batch, n_tokens, token_dim)
    time = jax.random.uniform(rng, (tokens.data.shape[0],))
    time_bc = time[:, None, None]

    # Compute flow path interpolation
    sigma_t = 1.0 - (1.0 - sigma_min) * time_bc
    theta_t = theta_0 * sigma_t + theta_data * time_bc

    # Compute target velocity
    # u_t = (theta - (1 - sigma_min) * theta_t) / (1 - sigma_t)
    numerator = theta_data - (1.0 - sigma_min) * theta_t
    denominator = 1.0 - (1.0 - sigma_min) * time_bc
    u_target = numerator / denominator

    # Set theta.data to interpolated values and evaluate vf
    new_data = tokens.data.at[:, tokens.partition_idx:].set(theta_t)
    theta_t_tokens = dataclasses.replace(tokens, data = new_data)
    v_pred = tfmpe.vf_network(
        theta_t_tokens,
        time
    )

    if tokens.padding_mask is not None:
        theta_padding_mask = tokens.padding_mask[:,tokens.partition_idx:,None]
        return jnp.sum(
            jnp.square(v_pred - u_target) * theta_padding_mask
        ) / jnp.sum(theta_padding_mask)

    return jnp.mean(
        jnp.square(v_pred - u_target)
    )
