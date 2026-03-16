"""Embedding layers for transformer architecture."""

from jax import numpy as jnp
from jax import random
from flax import nnx
from jaxtyping import Array
from ...preprocessing.tokens import Tokens


class GaussianFourierEmbedding(nnx.Module):
    """Gaussian Fourier feature embedding for continuous values.

    Maps input features through sin/cos of random Gaussian frequency
    basis to produce higher-dimensional embeddings.

    Attributes
    ----------
    b : nnx.Param
        Gaussian frequency basis matrix, shape (in_dim, out_dim // 2)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Gaussian Fourier embedding.

        Parameters
        ----------
        in_dim : int
            Input feature dimension
        out_dim : int
            Output feature dimension (must be even)
        rngs : Array
            JAX random key for initialization
        """
        b_dim = out_dim // 2
        self.b = nnx.Param(
            random.normal(rngs.params(), (in_dim, b_dim))
        )

    def __call__(self, inputs: Array) -> Array:
        """Apply Gaussian Fourier embedding.

        Computes concatenation of [cos(2π * inputs @ b),
        sin(2π * inputs @ b)].

        Parameters
        ----------
        inputs : Array
            Input array, shape (..., in_dim)

        Returns
        -------
        Array
            Embedded output, shape (..., out_dim)
        """
        x = 2 * jnp.pi * jnp.dot(inputs, self.b[...])
        return jnp.concatenate([
            jnp.cos(x),
            jnp.sin(x),
        ], axis=-1)


class Embedding(nnx.Module):
    """Embedding layer for token data.

    Combines value and label embeddings into a unified latent
    representation.

    Attributes
    ----------
    embedding : nnx.Embed
        Label embedding layer
    linear : nnx.Linear
        Linear projection to latent dimension
    functional_inputs_dim : int
        Dimension of functional inputs (0 if not used)
    """

    def __init__(
        self,
        value_dim: int,
        n_labels: int,
        label_dim: int,
        pos_dim: int,
        max_positions: int,
        latent_dim: int,
        rngs: nnx.Rngs,
        f_in_in_dim: int = 0,
        f_in_out_dim: int = 0,
        group_dim: int = 0,
        max_groups: int = 128,
    ) -> None:
        """Initialize Embedding layer.

        Parameters
        ----------
        value_dim : int
            Dimension of token values
        n_labels : int
            Number of distinct labels
        label_dim : int
            Embedding dimension for labels
        pos_dim : int
            Embedding dimension for within-group position
        max_positions : int
            Maximum number of within-group positions
        latent_dim : int
            Target latent dimension
        rngs : nnx.Rngs
            JAX random number generator for initialization
        f_in_in_dim : int, optional
            Input dimension of functional inputs (0 if not used).
            Default is 0.
        f_in_out_dim : int, optional
            Output dimension of functional input embeddings.
            Default is 0.
        group_dim : int, optional
            Embedding dimension for group id (0 to disable).
            Default is 0.
        max_groups : int, optional
            Maximum number of groups for group id embedding.
            Default is 128.
        """

        self.embedding = nnx.Embed(
            n_labels,
            features=label_dim,
            rngs=rngs,
        )

        self.pos_emb = nnx.Embed(
            max_positions,
            features=pos_dim,
            rngs=rngs,
        )

        self.group_dim = group_dim
        if group_dim > 0:
            self.group_emb = nnx.Embed(
                max_groups,
                features=group_dim,
                rngs=rngs,
            )

        if f_in_in_dim > 0:
            self.f_in_emb = GaussianFourierEmbedding(f_in_in_dim, f_in_out_dim, rngs)
        else:
            f_in_out_dim = 0

        # Input dimension: value + label + pos + time + group + functional_inputs
        in_dim = value_dim + label_dim + pos_dim + 1 + group_dim + f_in_out_dim

        self.linear = nnx.Linear(
            in_dim,
            latent_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        tokens: Tokens,
        time: Array
    ) -> Array:
        """Embed token data.

        Parameters
        ----------
        tokens : Tokens
            Token data object containing values, labels, and
            optional functional inputs

        Returns
        -------
        Array
            Embedded tokens, shape (*sample_shape, n_tokens,
            latent_dim)
        """
        # Extract components from Tokens
        values = tokens.data
        labels = tokens.labels
        functional_inputs = tokens.functional_inputs

        # Embed labels
        labels_emb = self.embedding(labels)

        # Extract sample and token shapes
        sample_shape = tokens.sample_shape
        n_tokens = values.shape[len(tokens.sample_shape)]

        # Expand labels
        labels_emb = jnp.broadcast_to(
            labels_emb,
            sample_shape + (n_tokens,) + (labels_emb.shape[-1],)
        )

        time_expanded = jnp.broadcast_to(
            jnp.atleast_1d(time)[..., None, None],
            sample_shape + (n_tokens, 1),
        )

        # Embed positions (integer indices)
        pos_emb = self.pos_emb(tokens.position.astype(jnp.int32))

        # Build concatenation
        parts = [
            values,
            labels_emb,
            pos_emb,
            time_expanded
        ]

        # Embed group_id (integer indices)
        if self.group_dim > 0:
            parts.append(
                self.group_emb(tokens.group_id.astype(jnp.int32))
            )

        if functional_inputs is not None:
            parts.append(
                self.f_in_emb(functional_inputs)
            )

        x = jnp.concatenate(parts, axis=-1)

        # Apply linear projection
        return self.linear(x)
