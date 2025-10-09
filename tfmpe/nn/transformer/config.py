"""Configuration for transformer architecture."""

from dataclasses import dataclass, field
from typing import Callable
from typing import Literal

from flax import nnx


@dataclass
class TransformerConfig:
    """Configuration container for transformer architecture parameters.

    This dataclass holds all configuration parameters needed to initialize
    a transformer model. It provides sensible defaults and validates that
    the latent dimension is divisible by the number of attention heads.

    Attributes
    ----------
    latent_dim : int
        Hidden dimension size for transformer layers
    n_encoder : int
        Number of encoder transformer blocks
    n_heads : int
        Number of attention heads in multi-head attention
    n_ff : int
        Number of sequential feedforward layers in MLP blocks
    label_dim : int
        Dimension for label embeddings
    label_dim : int
        Dimension for positional embeddings
    index_out_dim : int
        Output dimension for index embeddings
    dropout : float
        Dropout rate for regularization
    activation : Callable
        Activation function for feedforward layers (e.g., nnx.relu)

    Raises
    ------
    ValueError
        If latent_dim is not divisible by n_heads
    """

    latent_dim: int = 128
    n_encoder: int = 2
    n_heads: int = 4
    n_ff: int = 2
    label_dim: int = 32
    index_out_dim: int = 64
    pos_dim: int = 8
    dropout: float = 0.1
    attention: Literal['linear'] | Literal['softmax'] = 'softmax'
    activation: Callable = field(default_factory=lambda: nnx.relu)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises
        ------
        ValueError
            If latent_dim is not divisible by n_heads
        """
        if self.latent_dim % self.n_heads != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
