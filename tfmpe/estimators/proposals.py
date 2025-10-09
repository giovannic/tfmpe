from typing import Callable, Optional, Dict, Tuple
from jaxtyping import Array, PyTree

from jax import tree, numpy as jnp, random as jr

from .tfmpe import TFMPE
from ..preprocessing.utils import Labeller
from ..preprocessing.tokens import Tokens

def truncated_proposal_rejection(
    key: Array,
    model: TFMPE,
    labeller: Labeller,
    f_in: Dict[str, Array],
    n_samples: int,
    epsilon: float,
    y_obs: Dict[str, Array],
    prior_fn: Callable[[PyTree, int, int, dict], dict],
    n: int,
    prior_log_prob: Callable[[PyTree], float],
    prob_transform: Optional[Callable[[PyTree, Array], float]] = None,
    n_estimate: int =  10_000, #1_000_000,
    n_batch: Optional[int] = None,
    ) -> PyTree:
    """Sample truncated proposal via sampling importance resampling"""
    if n_batch is None:
        n_batch = n_samples
    estimate_key, key = jr.split(key)

    samples, log_prob = _batch_sample(
        estimate_key,
        model,
        labeller,
        prior_fn,
        n,
        y_obs,
        f_in,
        n_batch,
        n_estimate,
        prob_transform,
    )

    tau = jnp.quantile(log_prob, epsilon)
    m = jnp.zeros((0,))

    while jnp.sum(m) < n_samples:
        sample_key, key = jr.split(key)
        new_samples, new_log_prob = _batch_sample_prior(
            sample_key,
            model,
            labeller,
            prior_fn,
            n,
            y_obs,
            f_in,
            n_batch,
            n_batch,
            prob_transform,
        )
        samples = tree.map(
            lambda x, y: jnp.concatenate([x, y]),
            samples,
            new_samples
        )
        log_prob = jnp.concatenate([
            log_prob,
            new_log_prob
        ])
        m = jnp.concatenate([
            m,
            log_prob > tau
        ])

    samples = tree.map(
        lambda leaf: jnp.compress(m, leaf, axis=0)[:n_samples],
        samples
    )

    return samples

def truncated_proposal_sir(
    key: Array,
    model: TFMPE,
    labeller: Labeller,
    f_in: Optional[Dict[str, Array]],
    n_samples: int,
    epsilon: float,
    y_obs: Dict[str, Array],
    prior_fn: Callable,
    n: int,
    prior_log_prob: Callable[[PyTree], float],
    prob_transform: Optional[Callable[[PyTree, Array], float]] = None,
    n_estimate: int =  10_000, #1_000_000,
    n_batch: Optional[int] = None,
    ) -> PyTree:
    """Sample truncated proposal via sampling importance resampling"""
    if n_batch is None:
        n_batch = n_samples
    estimate_key, resample_key = jr.split(key)

    samples, log_prob = _batch_sample(
        estimate_key,
        model,
        labeller,
        prior_fn,
        n,
        y_obs,
        f_in,
        n_batch,
        n_estimate,
        prob_transform,
    )

    tau = jnp.quantile(log_prob, epsilon)
    m = log_prob > tau

    w = prior_log_prob(samples) - log_prob
    m = log_prob > tau

    indices = jr.categorical(
        resample_key,
        logits=jnp.extract(m, w),
        shape=(n_samples,),
        replace=True
    )

    samples = tree.map(
        lambda leaf: jnp.compress(m, leaf, axis=0)[indices],
        samples
    )

    return samples

def _batch_sample(
    key: Array,
    model: TFMPE,
    labeller: Labeller,
    prior_fn: Callable,
    n: int,
    y_obs: Dict[str, Array],
    f_in: Optional[Dict[str, Array]],
    n_batch: int,
    n_total: int,
    prob_transform: Optional[Callable[[PyTree, Array], float]] = None,
    ) -> Tuple[PyTree, Array]:
    """Estimate threshold for High Probability Region of approximate posterior"""
    samples = prior_fn(
        key,
        n,
        1,
        f_in
    )

    theta_template = tree.map(
        lambda leaf: jnp.zeros(
            (n_batch,) + leaf.shape[1:]
        ),
        samples
    )
    y_expanded = tree.map(
        lambda leaf: jnp.broadcast_to(
            leaf,
            (n_batch,) + leaf.shape[1:]
        ),
        y_obs
    )

    tokens, decoder = Tokens.from_pytree_with_decoder(
        {**y_expanded, **theta_template},
        condition=list(y_expanded.keys()),
        labeller=labeller,
        sample_ndims=1,
        functional_inputs=f_in,
    )

    n_sampled: int = 0
    all_tokens = None
    log_prob = jnp.array([])

    while True:
        output_tokens = model.sample_posterior(tokens)
        new_log_prob = model.log_prob_posterior_samples(output_tokens)

        if prob_transform is not None:
            new_log_prob = prob_transform(decoder(output_tokens), new_log_prob)

        log_prob = jnp.concatenate([log_prob, new_log_prob])

        if all_tokens is None:
            all_tokens = output_tokens
        else:
            all_tokens = tree.map(
                lambda x, y: jnp.concatenate([x, y]),
                all_tokens,
                output_tokens
            )

        n_sampled += n_batch

        if n_sampled >= n_total:
            break

    values = decoder(all_tokens)
    values = {
        k: v
        for k, v in values.items()
        if k in theta_template.keys()
    }

    return values, log_prob

def _batch_sample_prior(
    key: Array,
    model: TFMPE,
    labeller: Labeller,
    prior_fn: Callable,
    n: int,
    y_obs: Dict[str, Array],
    f_in: Dict[str, Array],
    n_batch: int,
    n_total: int,
    prob_transform: Optional[Callable[[PyTree, Array], float]] = None,
    ) -> Tuple[PyTree, Array]:
    """Estimate threshold for High Probability Region of approximate posterior"""
    samples, _ = prior_fn(
        key,
        n,
        1,
        f_in
    )

    theta_template = tree.map(
        lambda leaf: jnp.zeros(
            (n_batch,) + leaf.shape[1:]
        ),
        samples
    )
    y_expanded = tree.map(
        lambda leaf: jnp.broadcast_to(
            leaf,
            (n_batch,) + leaf.shape[1:]
        ),
        y_obs
    )

    _, decoder = Tokens.from_pytree_with_decoder(
        {**y_expanded, **theta_template},
        condition=list(y_expanded.keys()),
        labeller=labeller,
        sample_ndims=1,
        functional_inputs=f_in,
    )

    n_sampled: int = 0
    all_tokens = None
    log_prob = jnp.array([])

    while True:
        prior_samples = prior_fn(
            key,
            n,
            n_batch,
            f_in
        )
        output_tokens = Tokens.from_pytree(
            {**y_expanded, **prior_samples},
            condition=list(y_expanded.keys()),
            labeller=labeller,
            sample_ndims=1,
            functional_inputs=f_in,
        )
        new_log_prob = model.log_prob_posterior_samples(output_tokens)

        if prob_transform is not None:
            new_log_prob = prob_transform(decoder(output_tokens), new_log_prob)

        log_prob = jnp.concatenate([log_prob, new_log_prob])

        if all_tokens is None:
            all_tokens = output_tokens
        else:
            all_tokens = tree.map(
                lambda x, y: jnp.concatenate([x, y]),
                all_tokens,
                output_tokens
            )

        n_sampled += n_batch

        if n_sampled >= n_total:
            break

    values = decoder(all_tokens)
    values = {
        k: v
        for k, v in values.items()
        if k in theta_template.keys()
    }

    return values, log_prob
