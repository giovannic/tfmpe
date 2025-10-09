"""MLP model for TFMPE."""

import jax.numpy as jnp
from jaxtyping import Array
from flax import nnx

from ..preprocessing.tokens import Tokens


class MLP(nnx.Module):
    """Simple MLP for TFMPE.

    Processes flattened token data through fully connected layers
    to produce a vector field. Implements the TokenisedVectorField
    protocol for compatibility with TFMPE.

    Attributes
    ----------
    n_target_tokens : int
        Number of target tokens in output
    value_dim : int
        Dimension of token values
    layers : list
        List of hidden linear layers
    output_linear : nnx.Linear
        Output projection layer
    """

    def __init__(
        self,
        n_ff: int,
        latent_dim: int,
        tokens: Tokens,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize MLP.

        Parameters
        ----------
        n_ff : int
            Number of hidden layers
        latent_dim : int
            Width of hidden layers
        tokens : Tokens
            Example tokens to deduce input/output dimensions
        rngs : nnx.Rngs
            JAX random number generators
        """
        n_tokens = tokens.data.shape[-2]
        value_dim = tokens.data.shape[-1]
        n_target_tokens = n_tokens - tokens.partition_idx

        self.n_target_tokens = n_target_tokens
        self.value_dim = value_dim

        # Input: flattened tokens + time
        input_dim = n_tokens * value_dim + 1
        output_dim = n_target_tokens * value_dim

        # Build hidden layers
        layers = []
        in_dim = input_dim
        for _ in range(n_ff):
            layers.append(nnx.Linear(in_dim, latent_dim, rngs=rngs))
            in_dim = latent_dim
        self.layers = nnx.List(layers)

        self.output_linear = nnx.Linear(latent_dim, output_dim, rngs=rngs)

    def __call__(
        self,
        tokens: Tokens,
        time: Array,
        deterministic: bool = False,
    ) -> Array:
        """Forward pass through MLP.

        Parameters
        ----------
        tokens : Tokens
            Input tokens with data shape (*sample_shape, n_tokens, value_dim)
        time : Array
            Time values, scalar or shape (*sample_shape,)
        deterministic : bool, optional
            Unused, for API compatibility. Default is False.

        Returns
        -------
        Array
            Output vector field, shape (*sample_shape, n_target_tokens, value_dim)
        """
        del deterministic  # Unused

        # tokens.data: (*sample_shape, n_tokens, value_dim)
        data = tokens.data
        sample_shape = data.shape[:-2]

        # Flatten last two dims: (*sample_shape, n_tokens * value_dim)
        x = data.reshape(sample_shape + (-1,))

        # Broadcast time to (*sample_shape, 1)
        time_expanded = jnp.broadcast_to(
            jnp.atleast_1d(time)[..., None],
            sample_shape + (1,)
        )

        # Concatenate: (*sample_shape, n_tokens * value_dim + 1)
        x = jnp.concatenate([x, time_expanded], axis=-1)

        # Forward through hidden layers with ReLU
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)

        x = self.output_linear(x)

        # Reshape to (*sample_shape, n_target_tokens, value_dim)
        return x.reshape(sample_shape + (self.n_target_tokens, self.value_dim))
