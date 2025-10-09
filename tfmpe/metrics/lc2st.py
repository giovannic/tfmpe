from functools import partial
from typing import Tuple
from jax import random as jr, numpy as jnp, vmap
from jaxtyping import Array
from flax import nnx
import optax
from ..nn.classifier import (
    Classifier,
    MultiBinaryMLPClassifier,
    MultiFoldBinaryMLPClassifier,
    ce_loss
)

from ..nn.training import fit_nn
from dataclasses import dataclass

def _fit_classifier(
    seed: jnp.ndarray,
    classifier: Classifier, 
    u: jnp.ndarray,
    labels: jnp.ndarray,
    num_epochs=100,
    batch_size=100,
    optimizer: optax.GradientTransformation = optax.adam(0.0003),
    split=.9,
    delta=1e-2,
    patience=100
    ):

    n_split = int(u.shape[0] * split)

    data = {"data": {"u": u[:n_split], "y": labels[:n_split]}}
    val = {"data": {"u": u[n_split:], "y": labels[n_split:]}}
    
    opt = nnx.Optimizer(classifier, optimizer, wrt=nnx.Param)

    return fit_nn(
        classifier,
        train=data,
        val=val,
        opt=opt,
        loss=ce_loss,
        n_iter=num_epochs,
        batch_size=batch_size,
        rng=seed,
        delta=delta,
        patience=patience
    )

def _train_lc2st_classifiers(
    rng_key: Array,
    d_cal: Tuple[Array, Array, Array], 
    classifier: MultiFoldBinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
    num_epochs: int,
    batch_size: int = 100,
) -> None:
    """
    Local-Classifier 2 Sample Test – training both main and null classifiers.
    
    Input:
        rng_key: JAX PRNG key for reproducibility.
        d_cal: Calibration data tuple (x, theta, theta_q) where:
            - x: observations from p(x|theta)
            - theta: parameters from joint distribution p(theta, x)  
            - theta_q: parameters from estimated posterior q(theta|x)
        classifier: Main binary classifier for (x, theta) concatenated input
        null_classifier: Multiple null classifiers for permuted label training
        num_epochs: Number of epochs to train both classifiers.
        batch_size: Batch size for training.
    """
    x_cal, theta_cal, theta_q = d_cal
    N_cal = x_cal.shape[0]

    # Train main classifier
    # Class C=0: (x, theta) from joint distribution p(theta, x)
    # Class C=1: (x, theta_q) from posterior q(theta|x) p(x)
    u_joint = jnp.concatenate([x_cal, theta_cal], axis=-1)
    u_posterior = jnp.concatenate([x_cal, theta_q], axis=-1)
    u_main = jnp.concatenate([u_joint, u_posterior], axis=0)
    labels_main = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)

    rng_key, shuffle_key = jr.split(rng_key)
    shuffled_i = jr.permutation(shuffle_key, N_cal*2)
    u_main = u_main[shuffled_i]
    labels_main = labels_main[shuffled_i]

    n_ensemble = classifier.n
    n_folds = classifier.n_fold
    dim = u_main.shape[-1]
    
    # Implement folds
    shift = N_cal // n_folds
    if n_folds > 1:
        N_cal_main = N_cal - shift
    else:
        N_cal_main = N_cal

    @partial(vmap, in_axes=(0, None), out_axes=1)
    def fold(i, dataset):
        return jnp.roll(dataset, shift * i, axis=0)[:N_cal_main]

    u_main = fold(jnp.arange(n_folds), u_main)
    labels_main = fold(jnp.arange(n_folds), labels_main)

    # broadcast to ensembles
    u_main = jnp.broadcast_to(
        u_main[:, None, :, :],
        (N_cal_main, n_ensemble, n_folds, dim)
    )
    labels_main = jnp.broadcast_to(
        labels_main[:, None, :],
        (N_cal_main, n_ensemble, n_folds)
    )

    rng_key, main_key = jr.split(rng_key)
    _fit_classifier(
        main_key,
        classifier,
        u_main,
        labels_main,
        num_epochs=num_epochs,
        batch_size=batch_size,
        delta=1e-1,
        patience=100
    )

    # Train null classifiers with permuted labels
    n_null = null_classifier.n
    rng_key, null_key = jr.split(rng_key)

    # Create 3D input for null classifiers: (batch_size, n_classifiers, feature_dim)
    u_null_base = jnp.concatenate([u_joint, u_posterior], axis=0)  # (2*N_cal, feature_dim)
    u_null = jnp.broadcast_to(
        u_null_base[None, :, :], 
        (n_null, 2*N_cal, u_null_base.shape[-1])
    ).transpose(1, 0, 2)  # (2*N_cal, n_classifiers, feature_dim)

    # Generate different permuted labels for each null classifier
    base_labels = jnp.concatenate([jnp.zeros(N_cal), jnp.ones(N_cal)], axis=0)
    null_keys = jr.split(null_key, n_null)
    
    def permute_labels(key):
        return jr.permutation(key, base_labels)
    
    # Create permuted labels for each classifier
    permuted_labels = jnp.stack([
        permute_labels(key)
        for key in null_keys
    ], axis=1)  # (2*N_cal, n_classifiers)
    
    _fit_classifier(
        null_key,
        null_classifier,
        u_null,
        permuted_labels,
        num_epochs=num_epochs,
        batch_size=batch_size,
        delta=1e-1,
        patience=100
    )

def _evaluate_lc2st(
    observation: Array,
    posterior_samples: Array, 
    main_classifier: MultiFoldBinaryMLPClassifier,
    null_classifier: MultiBinaryMLPClassifier,
) -> Tuple[Array, Array]:
    """
    Local-Classifier 2 Sample Test evaluation in posterior space.
    
    Input:
        observation: The specific observation to evaluate consistency at.
        posterior_samples: Samples from estimated posterior q(theta|observation).
        main_classifier: Main classifier trained to distinguish joint vs posterior.
        null_classifier: Null classifiers trained with permuted labels.
    
    Output:
        null_test_statistics: Test statistics for each null classifier.
        t_mse_val: The calculated MSE test statistic for posterior samples.
        p_value: The p-value for the Local-Classifier 2 Sample Test.
    """
    main_classifier.eval()
    null_classifier.eval()
    n_samples = posterior_samples.shape[0]
    n_null = null_classifier.n
    
    # Create inputs for main classifier: (observation, posterior_samples)
    observation_broadcast = jnp.broadcast_to(
        observation[None, :], 
        (n_samples, observation.shape[0])
    )
    u_main = jnp.concatenate([observation_broadcast, posterior_samples], axis=-1)
    dim = u_main.shape[-1]

    # Broadcast u_main to n_ensembles, n_fold
    u_main_ens = jnp.broadcast_to(
        u_main[:, None, None, :],
        (n_samples, main_classifier.n, main_classifier.n_fold, dim)
    )

    # Compute main test statistic
    # t̂MSE = (1/n_samples) * sum((d_main(observation, theta_i) - 1/2)^2)
    d_main = main_classifier(u_main_ens)
    t_mse_val = jnp.mean((d_main - 0.5)**2)
    
    # Create 3D inputs for null classifiers
    u_null = jnp.broadcast_to(
        u_main[None, :, :],
        (n_null, n_samples, u_main.shape[-1])
    ).transpose(1, 0, 2)  # (n_samples, n_null, feature_dim)
    
    # Compute null test statistics
    # t̂null_h = (1/n_samples) * sum((d_null_h(observation, theta_i) - 1/2)^2)
    d_null = null_classifier(u_null)  # (n_samples, n_null)
    null_test_statistics = jnp.mean((d_null - 0.5)**2, axis=0)  # (n_null,)
    
    return null_test_statistics, t_mse_val

@dataclass
class LC2STResponse:
    main_stat: Array
    null_stats: Array

    @property
    def pvalue(self):
        x = jnp.mean(self.null_stats >= self.main_stat)
        x = jnp.maximum(x, 1. / (self.n_null + 1))
        return x

    @property
    def n_null(self):
        return self.null_stats.shape[0]

    def critical_value(self, alpha):
        return jnp.quantile(self.null_stats, alpha)


def run_lc2st(
    key: Array,
    d_cal: Tuple[Array, Array, Array],
    observation: Array,
    post: Array,
    latent_dim: int = 32,
    n_layers: int = 2,
    num_folds: int = 10,
    num_ensemble: int = 10,
    num_null: int = 100,
    n_epochs: int = 100,
    ) -> LC2STResponse:
    rngs = nnx.Rngs(key)
    dim = d_cal[0].shape[1] + d_cal[1].shape[1]
    main = MultiFoldBinaryMLPClassifier(
        dim=dim,
        latent_dim = latent_dim,
        n=num_ensemble,
        n_fold=num_folds,
        n_layers=n_layers,
        activation=nnx.relu,
        rngs=rngs,
    )

    null_classifier = MultiBinaryMLPClassifier(
        dim=dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        activation=nnx.relu,
        n=num_null,
        rngs=rngs,
    )

    # Train both classifiers together
    train_key, key = jr.split(key)
    _train_lc2st_classifiers(
        train_key,
        d_cal,
        main,
        null_classifier,
        n_epochs
    )

    null_stats, main_stat = _evaluate_lc2st(
        observation,
        post,
        main,
        null_classifier,
    )
    return LC2STResponse(
        main_stat = main_stat,
        null_stats = null_stats,
    )
