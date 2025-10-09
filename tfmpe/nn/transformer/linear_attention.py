"""Linear attention implementation."""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from flax.typing import Dtype, PrecisionLike
from flax.nnx.module import Module


def linear_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
) -> Array:
    """Basic linear attention (non-causal, parallel form).

    Implements: output = φ(Q)(φ(K)^T V) / (φ(Q) · Σφ(K))

    Uses ELU+1 as the feature map φ to ensure positive values.

    Parameters
    ----------
    query : Array
        Shape [batch..., q_length, num_heads, qk_depth_per_head]
    key : Array
        Shape [batch..., kv_length, num_heads, qk_depth_per_head]
    value : Array
        Shape [batch..., kv_length, num_heads, v_depth_per_head]
    mask : None
        Must be None. Linear attention does not support masking.
    dropout_rng : Array | None
        Unused, for interface compatibility.
    dropout_rate : float
        Unused, for interface compatibility.
    broadcast_dropout : bool
        Unused, for interface compatibility.
    deterministic : bool
        Unused, for interface compatibility.
    dtype : Dtype | None
        Unused, for interface compatibility.
    precision : PrecisionLike
        Precision for einsum operations.
    module : Module | None
        Unused, for interface compatibility.

    Returns
    -------
    Array
        Shape [batch..., q_length, num_heads, v_depth_per_head]
    """
    # Feature map: ELU + 1 (ensures positive values)
    def feature_map(x: Array) -> Array:
        return jax.nn.elu(x) + 1

    q = feature_map(query)
    k = feature_map(key)

    if mask is not None:
        q = q * mask[..., None]
        value = value * mask[..., None]

    # KV: φ(K)^T @ V -> [batch..., heads, d, v_d]
    kv = jnp.einsum('...lhd,...lhv->...hdv', k, value, precision=precision)

    # K sum: Σφ(K) -> [batch..., heads, d]
    k_sum = jnp.sum(k, axis=-3)

    # Output: φ(Q) @ KV -> [batch..., q_len, heads, v_d]
    output = jnp.einsum('...qhd,...hdv->...qhv', q, kv, precision=precision)

    # Normalizer: φ(Q) · Σφ(K) -> [batch..., q_len, heads]
    normalizer = jnp.einsum('...qhd,...hd->...qh', q, k_sum, precision=precision)
    normalizer = jnp.maximum(normalizer, 1e-6)

    return output / normalizer[..., None]
