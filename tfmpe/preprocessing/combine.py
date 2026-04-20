"""Combine multiple Tokens objects for dataset accumulation."""

import jax.numpy as jnp
from jax import lax, tree
from jaxtyping import Array
from typing import Optional
import dataclasses

from .tokens import Tokens

def combine_tokens(
    tokens1: Tokens, tokens2: Tokens, pad_to_even: bool = True
) -> Tokens:
    """
    Combine two Tokens objects by concatenating samples and padding tokens.

    Parameters
    ----------
    tokens1 : Tokens
        First Tokens object
    tokens2 : Tokens
        Second Tokens object

    Returns
    -------
    Tokens
        Combined Tokens with concatenated samples and padded tokens

    ValueError
        If tokens have incompatible functional_inputs or
        incompatible sample_ndims

    Notes
    -----
    - Sample dimension (axis 0) is concatenated
    - Token dimension (axis 1) is padded to max across both inputs
    - Padding tokens are applied such that token.partition_idx are consistent
    - Self-attention masks are padded with zeros
    - Padding masks track which tokens are real vs padded
    """
    # Check functional_inputs consistency
    has_func1 = tokens1.functional_inputs is not None
    has_func2 = tokens2.functional_inputs is not None
    if has_func1 != has_func2:
        raise ValueError(
            "Cannot combine tokens: one has functional_inputs, "
            "the other does not"
        )

    # If both have functional_inputs, check final dimension matches
    if has_func1 and has_func2:
        final_dim1 = tokens1.functional_inputs.shape[-1]  # type: ignore
        final_dim2 = tokens2.functional_inputs.shape[-1]  # type: ignore
        if final_dim1 != final_dim2:
            raise ValueError(
                "Cannot combine tokens: functional_inputs have different "
                f"final dimensions ({final_dim1} vs {final_dim2})"
            )

    if tokens1.sample_ndims != tokens2.sample_ndims:
        raise ValueError(
            "Cannot combine tokens with different sample_ndims:"
            f"sample_ndims ({tokens1.sample_ndims} vs {tokens2.sample_ndims})"
        )

    max_n_condition = max(
        tokens1.partition_idx,
        tokens2.partition_idx
    )

    max_n_target = max(
        tokens1.data.shape[tokens1.sample_ndims] - tokens1.partition_idx,
        tokens2.data.shape[tokens2.sample_ndims] - tokens2.partition_idx,
    )

    # Pad target dimension so total sequence length is even (cuDNN flash attn)
    if pad_to_even and (max_n_condition + max_n_target) % 2 == 1:
        max_n_target += 1

    sample_ndims = tokens1.sample_ndims

    # Add a basic padding mask (use replace to avoid mutating inputs)
    n_tokens1 = tokens1.data.shape[tokens1.sample_ndims]
    n_tokens2 = tokens2.data.shape[tokens2.sample_ndims]
    tokens1 = dataclasses.replace(
        tokens1,
        padding_mask=jnp.ones(tokens1.sample_shape + (n_tokens1,))
    )
    tokens2 = dataclasses.replace(
        tokens2,
        padding_mask=jnp.ones(tokens2.sample_shape + (n_tokens2,))
    )

    def split_leaf(leaf, idx, sample_ndims):
        context = lax.slice_in_dim(leaf, 0, idx, axis=sample_ndims)
        target = lax.slice_in_dim(leaf, idx, leaf.shape[sample_ndims], axis=sample_ndims)
        return context, target

    def pad_token_leaf(leaf1: Optional[Array], leaf2: Optional[Array]) -> Optional[Array]:
        if leaf1 is None or leaf2 is None:
            return None
        if max_n_condition == 0:
            return jnp.concatenate([
                _pad_data_to_max_tokens(leaf1, max_n_target, sample_ndims),
                _pad_data_to_max_tokens(leaf2, max_n_target, sample_ndims),
            ], axis=0)
        context_1, target_1 = split_leaf(leaf1, tokens1.partition_idx, sample_ndims)
        context_2, target_2 = split_leaf(leaf2, tokens2.partition_idx, sample_ndims)
        context = jnp.concatenate([
            _pad_data_to_max_tokens(context_1, max_n_condition, sample_ndims),
            _pad_data_to_max_tokens(context_2, max_n_condition, sample_ndims)
        ], axis=0)
        target = jnp.concatenate([
            _pad_data_to_max_tokens(target_1, max_n_target, sample_ndims),
            _pad_data_to_max_tokens(target_2, max_n_target, sample_ndims)
        ], axis=0)
        return jnp.concatenate([context, target], axis=sample_ndims)

    combined_tokens = tree.map(
        pad_token_leaf,
        tokens1,
        dataclasses.replace(tokens2, partition_idx = tokens1.partition_idx),
        is_leaf=lambda x: x is None
    )
    combined_tokens.partition_idx = max_n_condition
    return combined_tokens

def _pad_data_to_max_tokens(
    data: Array,
    max_n_tokens: int,
    sample_ndims: int,
    pad_value: float = 0.0,
) -> Array:
    """
    Pad data array to max_n_tokens on the token dimension.

    Parameters
    ----------
    data : Array
        Data array with shape (*sample_shape, n_tokens, max_batch_size)
    max_n_tokens : int
        Target number of tokens
    sample_ndims : int
        Number of sample dimensions
    pad_value : float
        Value to use for padding

    Returns
    -------
    Array
        Padded array with shape (*sample_shape, max_n_tokens,
        max_batch_size)
    """
    current_n_tokens = data.shape[sample_ndims]

    if current_n_tokens >= max_n_tokens:
        return data

    # Calculate padding: [(0, 0), ..., (0, pad_amount), (0, 0)]
    pad_width = [(0, 0)] * len(data.shape)
    pad_width[sample_ndims] = (0, max_n_tokens - current_n_tokens)

    return jnp.pad(
        data, pad_width, mode='constant', constant_values=pad_value
    )
