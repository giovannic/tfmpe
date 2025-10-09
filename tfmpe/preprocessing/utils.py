"""Utility functions for parameter processing."""

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple
from jaxtyping import Array

import jax.numpy as jnp

@dataclass
class Labeller:
    """
    Global label information for a set of keys.

    Stores key-to-label mappings and provides methods to generate
    label arrays for token configurations. This decouples labeling
    from the Tokens class, allowing independent token instances to
    share consistent label values through a shared Labeller.

    Attributes
    ----------
    label_map : Dict[str, int]
        Mapping from key names to integer label indices. Used to
        generate label arrays for token data. All values must be
        unique (no collisions allowed).
    """
    label_map: Dict[str, int]

    def __post_init__(self) -> None:
        """Validate label_map has no collisions."""
        values = self.label_map.values()
        if len(values) != len(set(values)):
            raise ValueError("label_map has collision: "
                             "multiple keys map to same label")

    @property
    def n_labels(self) -> int:
        """
        Get number of label classes.

        Returns the count of entries in label_map.

        Returns
        -------
        int
            Number of distinct keys in label_map
        """
        return len(self.label_map)

    @classmethod
    def for_keys(cls, keys: List[str]) -> 'Labeller':
        """
        Create Labeller with sequential labels for keys.

        Assigns sequential integer labels (0, 1, 2, ...) to keys
        in the order provided.

        Parameters
        ----------
        keys : List[str]
            List of keys to label in order

        Returns
        -------
        Labeller
            Labeller with sequential label assignments
        """
        label_map = {key: i for i, key in enumerate(keys)}
        return cls(label_map=label_map)

    def label(self, slices: Dict[str, 'SliceInfo']) -> Array:
        """
        Generate label array for given token configuration.

        Creates a 1D label array containing the label index for each
        token, ordered by slices.keys(). Each key contributes a number
        of labels equal to the total tokens for that key (product of
        event_shape dimensions).

        Parameters
        ----------
        slices : Dict[str, SliceInfo]
            Token metadata mapping with keys in slice dict order.
            Contains SliceInfo with event_shape for each key.

        Returns
        -------
        Array
            Label array with shape (n_total_tokens,) containing
            integer label indices from label_map.

        Raises
        ------
        ValueError
            If label_map is empty
        KeyError
            If a key in slices is not in label_map
        """
        if not self.label_map:
            raise ValueError("Labeller requires at least one key mapping")

        # Build labels for each key in slices dict order
        labels_list = []
        for key in slices.keys():
            if key not in self.label_map:
                raise KeyError(f"Key '{key}' not found in label_map")

            event_shape = slices[key].event_shape
            n_tokens = math.prod(event_shape)

            key_labels = jnp.full(n_tokens,
                                  self.label_map[key],
                                  dtype=jnp.int32)
            labels_list.append(key_labels)

        # Concatenate all labels in slice order
        return jnp.concatenate(labels_list)

class SliceInfo(NamedTuple):
    """
    Metadata for a slice in a flattened array.

    Attributes
    ----------
    offset : int
        Starting index in the flattened event dimension
    event_shape : Tuple[int, ...]
        Original event dimensions
    batch_shape : Tuple[int, ...]
        Original batch dimensions
    """
    offset: int
    event_shape: Tuple[int, ...]
    batch_shape: Tuple[int, ...]
