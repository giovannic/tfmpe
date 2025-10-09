from typing import Callable
from jaxtyping import PyTree, Array
from flax import nnx
from jax import numpy as jnp

class FFLayer(nnx.Module):

    activation: Callable

    def __init__(self, dim, activation, dropout_rate=0.5, rngs=nnx.Rngs(0)):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x

class BinaryMLPClassifier(nnx.Module):

    n_layers: int

    def __init__(
        self,
        dim,
        latent_dim,
        n_layers,
        activation,
        rngs=nnx.Rngs(0)
        ):

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs):
            return FFLayer(latent_dim, activation, rngs=rngs)

        self.layers = create_layer(rngs)
        self.n_layers = n_layers
        self.in_layer = nnx.Linear(dim, latent_dim, rngs=rngs)
        self.output = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, u):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, model):
            x = model(x)
            return x

        x = self.in_layer(u)
        x = forward(x, self.layers)
        x = self.output(x)[..., 0]
        return nnx.sigmoid(x)

class MultiBinaryMLPClassifier(nnx.Module):

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        n_layers: int,
        activation: Callable,
        n: int,
        rngs=nnx.Rngs(0)
        ):
        """
        dim: int the latent dimension
        n_layers: int number of layers
        activation: Callable activation function
        n: int number of classifiers to map over
        rngs: nnx.Rngs
        """
        @nnx.split_rngs(splits=n)
        @nnx.vmap
        def create_classifier(rngs):
            return BinaryMLPClassifier(dim, latent_dim, n_layers, activation, rngs=rngs)

        self.classifiers = create_classifier(rngs)
        self.n = n

    def __call__(self, u):
        assert u.ndim == 3, f"MultiBinaryMLPClassifier expects 3D input, got {u.ndim}D with shape {u.shape}"
        assert u.shape[1] == self.n, f"Second dimension must match number of classifiers ({self.n}), got shape {u.shape}"
        
        # Input shape is (batch_dim, n_classifiers, z_dim)
        # We want each classifier to process all batch samples for its corresponding slice
        @nnx.vmap(in_axes=(0, 1), out_axes=1)
        def call_classifier_on_slice(cls, u_slice):
            return cls(u_slice)

        return call_classifier_on_slice(self.classifiers, u)

class MultiFoldBinaryMLPClassifier(nnx.Module):

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        n_layers: int,
        activation: Callable,
        n: int,
        n_fold: int,
        rngs=nnx.Rngs(0)
        ):
        """
        dim: int the latent dimension
        n_layers: int number of layers
        activation: Callable activation function
        n: int number of classifiers to map over
        rngs: nnx.Rngs
        """
        @nnx.split_rngs(splits=n)
        @nnx.vmap
        def create_classifier(rngs):
            return BinaryMLPClassifier(dim, latent_dim, n_layers, activation, rngs=rngs)

        self.classifiers = create_classifier(rngs)
        self.n = n
        self.n_fold = n_fold

    def __call__(self, u):
        assert u.ndim == 4, f"MultiFoldBinaryMLPClassifier expects 4D input, got {u.ndim}D with shape {u.shape}"
        assert u.shape[1] == self.n, f"Second dimension must match number of classifiers ({self.n}), got shape {u.shape}"
        assert u.shape[2] == self.n_fold, f"Third dimension must match number of folds ({self.n_fold}), got shape {u.shape}"
        
        # Input shape is (batch_dim, n_classifiers, z_dim)
        # We want each classifier to process all batch samples for its corresponding slice
        @nnx.vmap(in_axes=(0, 1), out_axes=1)
        @nnx.vmap(in_axes=(None, 1), out_axes=1)
        def call_classifier_on_slice(cls, u_slice):
            return cls(u_slice)

        return call_classifier_on_slice(self.classifiers, u)

Classifier = MultiBinaryMLPClassifier | BinaryMLPClassifier | MultiFoldBinaryMLPClassifier

def ce_loss(
    model: Classifier,
    batch: PyTree,
    _: jnp.ndarray
    ) -> Array:
    """Calculates binary cross-entropy loss for the classifier."""
    preds = model(batch['data']['u'])
    labels = batch['data']['y']
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7) # Clip to avoid log(0)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
    return loss
