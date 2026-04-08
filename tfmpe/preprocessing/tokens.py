"""Unified parameter data interface.

Provides the Tokens class for consolidating structured parameter handling,
masking, functional inputs, and metadata into a single coherent interface.
"""

from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Callable,
)
from math import prod

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array

from .flatten import flatten_pytree
from .utils import Labeller
from .functional_inputs import flatten_functional_inputs
from .reconstruct import decode_pytree

@register_pytree_node_class
@dataclass
class Tokens:
    """
    Unified container for structured token data.

    Stores all parameters and observations in a single flattened array,
    with metadata for efficient decoding to structured format.

    Attributes
    ----------
    data : Array
        All flattened tokens, shape (*sample_shape, n_total_tokens,
        max_batch_size)
    labels : Array
        Integer labels per token, shape (*sample_shape, n_total_tokens)
    position : Array
        Position index per token within its key group,
        shape (*sample_shape, n_total_tokens)
    condition : Array
        Conditioning indicator per token (1.0 for conditioning,
        0.0 for target), shape (*sample_shape, n_total_tokens)
    partition_idx: int
        Static index which separates conditional and target data
    padding_mask : Optional[Array]
        Padding mask, shape (*sample_shape, n_total_tokens)
    functional_inputs : Optional[Array]
        Functional inputs for tokens, shape (*sample_shape,
        n_total_tokens, max_batch_size)
    """

    data: Array
    labels: Array
    position: Array
    condition: Array
    partition_idx: int
    padding_mask: Optional[Array]
    functional_inputs: Optional[Array]
    group_id: Array

    @property
    def sample_ndims(self) -> int:
        """
        Get number of sample dimensions from data array.

        Returns
        -------
        int
            Number of leading sample dimensions
        """
        return len(self.data.shape) - 2  # Remove event and batch

    @property
    def sample_shape(self) -> Tuple[int, ...]:
        """
        Get sample shape from data array.

        Returns
        -------
        Tuple[int, ...]
            Shape of sample dimensions
        """
        return self.data.shape[:self.sample_ndims]

    @classmethod
    def from_pytree(
        cls,
        data: Dict[str, Array],
        condition: List[str],
        labeller: Optional[Labeller] = None,
        functional_inputs: Optional[Dict[str, Array]] = None,
        sample_ndims: int = 1,
        batch_ndims: Optional[Dict[str, int]] = None,
        pad_to_even: bool = True,
    ) -> 'Tokens':
        """
        Create Tokens from structured PyTree.

        All keys in data are flattened into a single token array.

        Parameters
        ----------
        data: Dict[str, Array]
            Dictionary of model variable samples. Each array should have shape
            (*sample_dims, *event_dims, *batch_dims).
        condition: List[str]
            List of keys which correspond to conditioning variables
        labeller : Optional[Labeller], optional
            Labeller instance for generating consistent labels across tokens.
            If None, creates a default labeller with sequential indices.
        functional_inputs : Optional[Dict[str, Array]], optional
            Dictionary of functional input arrays matching data structure
        sample_ndims : int, optional
            Number of leading sample dimensions. Default is 1.
        batch_ndims : Optional[Dict[str, int]], optional
            Number of trailing batch dimensions for each key.
            If None, defaults to 1 for all keys.
        return_decoder: Optional[bool], optional
            Whether to return a decoding function

        Returns
        -------
        Tokens
            Token object
        """
        tokens, _ = cls._from_pytree_impl(
            data, condition, labeller, functional_inputs,
            sample_ndims, batch_ndims, pad_to_even
        )
        return tokens

    @classmethod
    def from_pytree_with_decoder(
        cls,
        data: Dict[str, Array],
        condition: List[str],
        labeller: Optional[Labeller] = None,
        functional_inputs: Optional[Dict[str, Array]] = None,
        sample_ndims: int = 1,
        batch_ndims: Optional[Dict[str, int]] = None,
        pad_to_even: bool = True,
    ) -> Tuple['Tokens', Callable[['Tokens'], Dict[str, Array]]]:
        """
        Create Tokens from structured PyTree with a decoder function.

        All keys in data are flattened into a single token array.

        Parameters
        ----------
        data: Dict[str, Array]
            Dictionary of model variable samples. Each array should have shape
            (*sample_dims, *event_dims, *batch_dims).
        condition: List[str]
            List of keys which correspond to conditioning variables
        labeller : Optional[Labeller], optional
            Labeller instance for generating consistent labels across tokens.
            If None, creates a default labeller with sequential indices.
        functional_inputs : Optional[Dict[str, Array]], optional
            Dictionary of functional input arrays matching data structure
        sample_ndims : int, optional
            Number of leading sample dimensions. Default is 1.
        batch_ndims : Optional[Dict[str, int]], optional
            Number of trailing batch dimensions for each key.
            If None, defaults to 1 for all keys.

        Returns
        -------
        Tuple[Tokens, Callable[[Tokens], Dict[str, Array]]]
            Token object and decoding function
        """
        return cls._from_pytree_impl(
            data, condition, labeller, functional_inputs,
            sample_ndims, batch_ndims, pad_to_even
        )

    @classmethod
    def _from_pytree_impl(
        cls,
        data: Dict[str, Array],
        condition: List[str],
        labeller: Optional[Labeller],
        functional_inputs: Optional[Dict[str, Array]],
        sample_ndims: int,
        batch_ndims: Optional[Dict[str, int]],
        pad_to_even: bool = True,
    ) -> Tuple['Tokens', Callable[['Tokens'], Dict[str, Array]]]:
        """
        Internal implementation for from_pytree methods.

        Flattens the input PyTree into a token array, generates labels
        and position indices, and creates a decoder closure that can
        reconstruct the original structure.

        Parameters
        ----------
        data : Dict[str, Array]
            Dictionary of parameter arrays to flatten
        condition : List[str]
            Keys corresponding to conditioning variables (placed first)
        labeller : Optional[Labeller]
            Label generator, or None to create default sequential labels
        functional_inputs : Optional[Dict[str, Array]]
            Optional functional inputs to flatten alongside data
        sample_ndims : int
            Number of leading sample dimensions
        batch_ndims : Optional[Dict[str, int]]
            Batch dimensions per key, or None to default to 1

        Returns
        -------
        Tuple[Tokens, Callable[[Tokens], Dict[str, Array]]]
            The constructed Tokens object and a decoder function that
            reconstructs the original PyTree structure from token data
        """
        # Default batch_ndims to 1 for all keys
        if batch_ndims is None:
            batch_ndims = {key: 1 for key in data.keys()}

        # Create default labeller if not provided
        if labeller is None:
            labeller = Labeller.for_keys(list(data.keys()))

        # Flatten the PyTree
        # Sort data such that conditioning variables come first
        key_order = sorted(data.keys(), key=lambda k: k not in condition)
        data = { k: data[k] for k in key_order }
        flat_data, slices = flatten_pytree(
            data,
            sample_ndims=sample_ndims,
            batch_ndims=batch_ndims
        )
        partition_idx = next(
            s.offset
            for k, s in slices.items()
            if k not in condition
        )

        # Build labels array
        total_tokens = flat_data.shape[sample_ndims]
        sample_shape = flat_data.shape[:sample_ndims]

        # Generate labels using Labeller
        labels_1d = labeller.label(slices)
        broadcast_shape = sample_shape + (total_tokens,)
        labels = jnp.broadcast_to(
            labels_1d.reshape((1,) * sample_ndims + (total_tokens,)),
            broadcast_shape
        )

        # Flatten functional inputs if provided
        func_inputs_flat = None
        if functional_inputs is not None:
            func_inputs_flat = flatten_functional_inputs(
                functional_inputs,
                slices,
                sample_ndims=sample_ndims
            )
            
        # Compute within-group position and group_id
        position_parts = []
        group_id_parts = []
        for s in slices.values():
            n_tokens_key = prod(s.event_shape)
            if len(s.event_shape) >= 2:
                n_groups = s.event_shape[0]
                inner_dim = prod(s.event_shape[1:])
                position_parts.append(
                    jnp.tile(jnp.arange(inner_dim), n_groups)
                )
                group_id_parts.append(
                    jnp.repeat(jnp.arange(n_groups), inner_dim)
                )
            else:
                position_parts.append(jnp.arange(n_tokens_key))
                group_id_parts.append(jnp.zeros(n_tokens_key))

        position = jnp.concatenate(position_parts)
        position = jnp.broadcast_to(
            position.reshape((1,) * sample_ndims + (total_tokens,)),
            broadcast_shape
        )

        group_id = jnp.concatenate(group_id_parts)
        group_id = jnp.broadcast_to(
            group_id.reshape((1,) * sample_ndims + (total_tokens,)),
            broadcast_shape
        )

        condition_values = jnp.concatenate([
            jnp.full(
                (prod(v.event_shape),),
                int(k in condition),
                dtype=float
            )
            for k, v in slices.items()
        ])
        condition_values = jnp.broadcast_to(
            condition_values.reshape((1,) * sample_ndims + (total_tokens,)),
            broadcast_shape
        )

        padding_mask = None

        # Pad to even sequence length for cuDNN flash attention
        if pad_to_even and total_tokens % 2 == 1:
            padding_mask = jnp.ones(broadcast_shape)
            pad_1d = [(0, 0)] * len(broadcast_shape)
            pad_1d[sample_ndims] = (0, 1)

            labels = jnp.pad(labels, pad_1d, constant_values=0)
            position = jnp.pad(position, pad_1d, constant_values=0)
            condition_values = jnp.pad(
                condition_values, pad_1d, constant_values=0
            )
            group_id = jnp.pad(group_id, pad_1d, constant_values=0)
            padding_mask = jnp.pad(padding_mask, pad_1d, constant_values=0)

            pad_data = [(0, 0)] * len(flat_data.shape)
            pad_data[sample_ndims] = (0, 1)
            flat_data = jnp.pad(flat_data, pad_data, constant_values=0)

            if func_inputs_flat is not None:
                pad_fi = [(0, 0)] * len(func_inputs_flat.shape)
                pad_fi[sample_ndims] = (0, 1)
                func_inputs_flat = jnp.pad(
                    func_inputs_flat, pad_fi, constant_values=0
                )

        tokens = cls(
            data=flat_data,
            labels=labels,
            position=position,
            condition=condition_values,
            padding_mask=padding_mask,
            functional_inputs=func_inputs_flat,
            group_id=group_id,
            partition_idx=partition_idx
        )

        # Capture original token count for decoder to strip padding
        orig_total_tokens = total_tokens

        def decoder(tokens: 'Tokens') -> Dict[str, Array]:
            data = tokens.data
            if data.shape[tokens.sample_ndims] > orig_total_tokens:
                data = data[
                    tuple(
                        slice(None) if i != tokens.sample_ndims
                        else slice(0, orig_total_tokens)
                        for i in range(len(data.shape))
                    )
                ]
            return decode_pytree(
                data,
                slices,
                tokens.sample_shape,
                is_subset=False
            )

        return tokens, decoder

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        """
        Flatten Tokens for JAX PyTree operations.

        Returns
        -------
        Tuple[Tuple, Dict[str, Any]]
            (children, aux_data) where children are arrays with sample
            dimension that get transformed by tree.map, and aux_data
            contains static metadata
        """
        children = (
            self.data,
            self.labels,
            self.position,
            self.condition,
            self.padding_mask,
            self.functional_inputs,
            self.group_id,
        )
        aux_data = {"partition_idx": self.partition_idx}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Dict[str, Any],
        children: Tuple
    ) -> 'Tokens':
        """
        Unflatten Tokens from JAX PyTree operations.

        Parameters
        ----------
        aux_data : Dict[str, Any]
            Static metadata
        children : Tuple
            Arrays with sample dimension

        Returns
        -------
        Tokens
            Reconstructed Tokens object
        """
        (
            data,
            labels,
            position,
            condition,
            padding_mask,
            functional_inputs,
            group_id,
        ) = children
        return cls(
            data=data,
            labels=labels,
            position=position,
            condition=condition,
            padding_mask=padding_mask,
            functional_inputs=functional_inputs,
            group_id=group_id,
            partition_idx=aux_data["partition_idx"]
        )
