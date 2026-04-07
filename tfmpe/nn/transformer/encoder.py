"""Encoder and decoder blocks for transformer architecture."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention

from .config import TransformerConfig
from .linear_attention import linear_attention


def cudnn_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
    **kwargs,
) -> Array:
    """Flash attention via cuDNN backend.

    Thin wrapper around jax.nn.dot_product_attention that forces
    the cuDNN implementation and discards Flax-specific kwargs
    (dropout_rng, dropout_rate, etc.).
    """
    if mask is not None:
        mask = mask.astype(jnp.bool_)
    return jax.nn.dot_product_attention(
        query, key, value, mask=mask, implementation="cudnn"
    )

class FFLayer(nnx.Module):
    """Single feedforward layer with linear, dropout, and activation.

    Attributes
    ----------
    linear : nnx.Linear
        Linear transformation layer
    dropout : nnx.Dropout
        Dropout regularization
    activation : Callable
        Activation function (e.g., nnx.relu)
    """

    activation: Callable

    def __init__(
        self,
        dim: int,
        dropout: float,
        activation: Callable,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        """Initialize feedforward layer.

        Parameters
        ----------
        dim : int
            Feature dimension (input and output)
        dropout : float
            Dropout rate
        activation : Callable
            Activation function
        rngs : nnx.Rngs
            Random number generator state
        dtype : jnp.dtype
            Dtype for linear layer parameters
        """
        self.linear = nnx.Linear(dim, dim, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        """Apply feedforward transformation.

        Parameters
        ----------
        x : Array
            Input array of shape (..., dim)

        Returns
        -------
        Array
            Output array of shape (..., dim)
        """
        x = self.linear(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class MLP(nnx.Module):
    """Multi-layer feedforward network.

    Applies sequential feedforward layers with dropout and activation
    functions. Maintains input shape through the network.

    Attributes
    ----------
    n_layers : int
        Number of feedforward layers
    layers : nnx.Module
        Vmapped FFLayer modules
    """

    n_layers: int

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize MLP network.

        Parameters
        ----------
        config : TransformerConfig
            Configuration containing latent_dim, n_ff, dropout, activation
        rngs : nnx.Rngs
            Random number generator state
        """
        n_ff = config.n_ff
        dim = config.latent_dim
        dropout = config.dropout
        activation = config.activation
        ops_dtype = config.ops_dtype

        @nnx.split_rngs(splits=n_ff)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs) -> FFLayer:
            """Create a single feedforward layer."""
            return FFLayer(dim, dropout, activation, rngs=rngs, dtype=ops_dtype)

        self.layers = create_layer(rngs)
        self.n_layers = n_ff

    def __call__(self, x: Array) -> Array:
        """Apply MLP transformation.

        Sequentially applies each feedforward layer, preserving input
        shape.

        Parameters
        ----------
        x : Array
            Input array of shape (..., latent_dim)

        Returns
        -------
        Array
            Output array of shape (..., latent_dim)
        """

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        @nnx.remat(
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        )
        def forward(
            x: Array,
            model: FFLayer,
        ) -> Array:
            """Apply a single layer and return updated state."""
            x = model(x)
            return x

        return forward(x, self.layers)


class EncoderBlock(nnx.Module):
    """Self-attention transformer encoder block.

    Applies multi-head self-attention followed by feedforward network,
    with residual connections and layer normalization after each
    sub-layer.

    Attributes
    ----------
    attention : nnx.MultiHeadAttention
        Multi-head self-attention module
    att_norm : nnx.LayerNorm
        Layer normalization after attention
    mlp : MLP
        Feedforward network
    ff_norm : nnx.LayerNorm
        Layer normalization after feedforward
    """

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize encoder block.

        Parameters
        ----------
        config : TransformerConfig
            Configuration containing latent_dim, n_heads, dropout
        rngs : nnx.Rngs
            Random number generator state
        """
        latent_dim = config.latent_dim
        n_heads = config.n_heads

        if config.attention == 'softmax':
            attention_fn = dot_product_attention
        elif config.attention == 'linear':
            attention_fn = linear_attention
        elif config.attention == 'cudnn':
            attention_fn = cudnn_attention
        else:
            raise ValueError(
                "TransformerConfig specifies unknown attention: {config.attention}"
            )

        self.ops_dtype = config.ops_dtype
        self.sensitive_ops_dtype = config.sensitive_ops_dtype

        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=latent_dim,
            qkv_features=latent_dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.dropout,
            decode=False,
            dtype=config.sensitive_ops_dtype,
            attention_fn=attention_fn,
            rngs=rngs,
        )
        self.att_norm = nnx.LayerNorm(
            num_features=latent_dim,
            dtype=config.sensitive_ops_dtype,
            rngs=rngs,
        )
        self.mlp = MLP(config=config, rngs=rngs)
        self.ff_norm = nnx.LayerNorm(
            num_features=latent_dim,
            dtype=config.sensitive_ops_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        deterministic: bool = False,
    ) -> Array:
        """Apply encoder block transformation.

        Applies self-attention with residual connection and layer
        normalization, followed by feedforward with residual connection
        and layer normalization.

        Parameters
        ----------
        x : Array
            Input array of shape (..., latent_dim)
        mask : Optional[Array]
            Unused. Linear attention does not support masking.
        deterministic : bool
            If True, disable dropout for deterministic inference.

        Returns
        -------
        Array
            Output array of shape (..., latent_dim)
        """
        # Self-attention with residual and norm
        x_op = x.astype(self.ops_dtype)
        attn_out = self.attention(
            x_op, x_op, x_op, mask=mask, deterministic=deterministic
        )
        x = x.astype(self.sensitive_ops_dtype) + attn_out.astype(self.sensitive_ops_dtype)
        x = self.att_norm(x)

        # Feedforward with residual and norm
        ff_out = self.mlp(x.astype(self.ops_dtype))
        x = x.astype(self.sensitive_ops_dtype) + ff_out.astype(self.sensitive_ops_dtype)
        x = self.ff_norm(x)

        # Cast back to ops_dtype for scan carry compatibility
        return x.astype(self.ops_dtype)
