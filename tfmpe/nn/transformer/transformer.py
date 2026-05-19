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
    """Encoder-only transformer for TFMPE.

    Encodes all tokens through self-attention encoder blocks, then
    extracts and projects target tokens to produce a vector field.

    Attributes
    ----------
    config : TransformerConfig
        Configuration for transformer architecture
    embedding : Embedding
        Embedding layer for token data
    encoder_blocks : nnx.Module
        Vmapped encoder blocks
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
            Configuration containing latent_dim, n_encoder,
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
            ops_dtype=config.ops_dtype,
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
            dtype=config.ops_dtype,
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
            if self.config.attention == 'cudnn':
                # Pass (batch,) seq lengths — varlen flash attention, no dense mask
                mask = jnp.sum(tokens.padding_mask, axis=-1).astype(jnp.int32)
            else:
                # Compact key-padding mask (B, 1, 1, L); XLA broadcasts internally
                mask = tokens.padding_mask[:, None, None, :]

        else:
            mask = None

        # Apply encoder blocks sequentially via scan
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        @nnx.remat(
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        )
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
        tokens: Tokens
            tokens to encode
        time : Array
            Time values, shape (*sample_shape,) or (*sample_shape, 1)
        deterministic : bool, optional
            If True, disable dropout. Default is False.

        Returns
        -------
        Array
            Output vector field for target tokens
        """
        output = self.encode(
            tokens=tokens,
            time=time,
            deterministic=deterministic,
        )

        return self.output_linear(output)
