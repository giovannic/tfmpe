"""Encoder and decoder blocks for transformer architecture."""

from typing import Callable, Optional

from jaxtyping import Array
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention

from .config import TransformerConfig
from .linear_attention import linear_attention

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
        """
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
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

        @nnx.split_rngs(splits=n_ff)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs) -> FFLayer:
            """Create a single feedforward layer."""
            return FFLayer(dim, dropout, activation, rngs=rngs)

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
        else:
            raise ValueError(
                "TransformerConfig specifies unknown attention: {config.attention}"
            )

        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=latent_dim,
            qkv_features=latent_dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.dropout,
            decode=False,
            attention_fn=attention_fn,
            rngs=rngs,
        )
        self.att_norm = nnx.LayerNorm(
            num_features=latent_dim,
            dtype=float,
            rngs=rngs,
        )
        self.mlp = MLP(config=config, rngs=rngs)
        self.ff_norm = nnx.LayerNorm(
            num_features=latent_dim,
            dtype=float,
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
        attn_out = self.attention(
            x, x, x, mask=mask, deterministic=deterministic
        )
        x = x + attn_out
        x = self.att_norm(x)

        # Feedforward with residual and norm
        ff_out = self.mlp(x)
        x = x + ff_out
        x = self.ff_norm(x)

        return x
