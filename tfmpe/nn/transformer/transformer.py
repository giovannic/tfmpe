"""Main transformer model for TFMPE."""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from flax import nnx

from .config import TransformerConfig
from .embedding import Embedding
from .encoder import EncoderBlock
from ...preprocessing.tokens import Tokens


class Transformer(nnx.Module):
    """Encoder only transformer for TFMPE.

    Processes context and parameter tokens through separate encoder
    blocks with shared attention, then decoder blocks with
    cross-attention.

    Attributes
    ----------
    config : TransformerConfig
        Configuration for transformer architecture
    embedding : Embedding
        Embedding layer for token data
    encoder_blocks : nnx.Module
        Vmapped encoder blocks
    decoder_blocks : nnx.Module
        Vmapped decoder blocks
    output_linear : nnx.Linear
        Linear layer projecting from latent_dim to value_dim
    """

    config: TransformerConfig
    value_dim: int

    def __init__(
        self,
        config: TransformerConfig,
        tokens: Tokens,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize transformer.

        Deduces value_dim, n_labels, and functional_inputs_dim from
        tokens.

        Parameters
        ----------
        config : TransformerConfig
            Configuration containing latent_dim, n_encoder, n_decoder,
            n_heads, n_ff, label_dim, index_out_dim, dropout,
            activation
        tokens : Tokens
            Full Tokens object containing all data
        rngs : nnx.Rngs
            JAX random number generators for parameter
            initialization
        """
        self.config = config
        self.value_dim = tokens.data.shape[-1]

        n_labels = jnp.unique(tokens.labels).shape[0]
        f_in_in_dim = (
            tokens.functional_inputs.shape[-1]
            if tokens.functional_inputs is not None
            else 0
        )

        # Create embedding layer
        self.embedding = Embedding(
            value_dim=self.value_dim,
            n_labels=n_labels,
            label_dim=config.label_dim,
            pos_dim=config.pos_dim,
            max_positions=config.max_positions,
            latent_dim=config.latent_dim,
            rngs=rngs,
            f_in_in_dim=f_in_in_dim,
            f_in_out_dim=config.index_out_dim,
            group_dim=config.group_dim,
            max_groups=config.max_groups,
        )

        # Create encoder blocks via vmap
        @nnx.split_rngs(splits=config.n_encoder)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_encoder_block(rngs: nnx.Rngs) -> EncoderBlock:
            """Create a single encoder block."""
            return EncoderBlock(config=config, rngs=rngs)

        self.encoder_blocks = create_encoder_block(rngs)

        # Create output linear layer
        self.output_linear = nnx.Linear(
            config.latent_dim,
            self.value_dim,
            rngs=rngs,
        )

    def encode(
        self,
        tokens: Tokens,
        time: Array,
        deterministic: bool = False,
    ) -> Array:
        """Encode tokens through encoder blocks.

        Parameters
        ----------
        tokens : Tokens
            Input tokens to encode
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Encoded tokens, shape (*sample_shape, n_tokens,
            latent_dim)
        """
        # Embed tokens
        x = self.embedding(tokens, time)

        # Apply encoder blocks sequentially via scan
        if tokens.padding_mask is not None:
            sample_shape = tokens.sample_shape
            n_tokens = tokens.data.shape[tokens.sample_ndims]
            n_heads = self.config.n_heads
            mask = tokens.padding_mask[:, None, None, :]
            mask = jnp.broadcast_to(mask, sample_shape + (n_heads, n_tokens, n_tokens))
        else:
            mask = None

        # Apply encoder blocks sequentially via scan
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(
            x: Array,
            encoder_block: EncoderBlock,
        ) -> Array:
            """Apply a single encoder block and return updated state."""
            x = encoder_block(
                x,
                mask=mask,
                deterministic=deterministic,
            )
            return x

        x = forward(x, self.encoder_blocks)

        token_dim = len(tokens.sample_shape)
        return jax.lax.slice_in_dim(
            x,
            tokens.partition_idx,
            None,
            axis=token_dim
        )

    def __call__(
        self,
        tokens: Tokens,
        time: Array,
        deterministic: bool = False,
    ) -> Array:
        """Forward pass through transformer.

        Parameters
        ----------
        context : Tokens
            Context tokens to encode
        params : Tokens
            Parameter tokens to decode
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Output vector field, shape matching params.data
        """
        # Encode context
        output = self.encode(
            tokens=tokens,
            time=time,
            deterministic=deterministic,
        )

        return self.output_linear(output)
