"""Tokenized Flow Matching Posterior Estimator (TFMPE)."""

from typing import Optional, Protocol

import diffrax
import jax
import jax.numpy as jnp
import dataclasses
from flax import nnx
from jaxtyping import Array

from ..preprocessing.tokens import Tokens
from .ode import (
    solve_forward_ode,
    solve_augmented_ode
)

class TokenisedVectorField(Protocol):
    """Protocol for vector field networks on tokenised data.

    A vector field network maps (context, params, time) -> velocity.
    Must be callable (typically an nnx.Module) to work with NNX
    transformations.
    """

    def __call__(
        self, tokens: Tokens, time: Array
    ) -> Array:
        """Compute velocity for flow matching.

        Parameters
        ----------
        tokens: Tokens
            Tokens to compute vector fields for
        time : Array
            Time points, shape (batch,) or scalar

        Returns
        -------
        Array
            Velocity prediction, same shape as params.data
        """
        ...


class BaseDistribution(Protocol):
    """Protocol for base distributions in flow matching.

    A base distribution must support sampling and log probability
    evaluation. Sampling is stateful and managed by nnx.Rngs.
    """

    def sample(self, shape: tuple) -> Array:
        """Sample from the base distribution.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate

        Returns
        -------
        Array
            Samples with given shape
        """
        ...

    def log_prob(self, x: Array) -> Array:
        """Compute log probability of the base distribution.

        Parameters
        ----------
        x : Array
            Values to evaluate

        Returns
        -------
        Array
            Log probabilities
        """
        ...


class NormalDistribution(nnx.Module):
    """Standard normal distribution base distribution.

    Provides sample() and log_prob() methods for base distribution
    in flow matching. Implemented as nnx.Module for JAX compilation
    compatibility.

    Attributes
    ----------
    rngs : nnx.Rngs
        RNG streams for stochastic sampling
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize NormalDistribution with RNG streams.

        Parameters
        ----------
        rngs : nnx.Rngs
            RNG streams for sampling. Will use default stream
            accessed via rngs() or rngs.params()
        """
        self.rngs = rngs

    def sample(self, shape: tuple) -> Array:
        """Sample from standard normal distribution.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate

        Returns
        -------
        Array
            Samples from N(0, I) with given shape
        """
        rng = self.rngs.params()
        return jax.random.normal(rng, shape)

    def log_prob(self, x: Array) -> Array:
        """Compute log probability of standard normal.

        Parameters
        ----------
        x : Array
            Values to evaluate

        Returns
        -------
        Array
            Sum of log probabilities over all dimensions
        """
        return jnp.sum(jax.scipy.stats.norm.logpdf(x))


class TFMPE(nnx.Module):
    """Unified estimator for sampling and log probability computation.

    Combines ODE solving for forward (sampling) and backward
    (log probability) trajectories using a trained vector field
    network.

    Attributes
    ----------
    vf_network : TokenisedVectorField
        Vector field network f(context, params, t) -> Array.
        Must be an nnx.Module so its state (including RNG
        streams) is properly managed during transformations.
    base_dist : nnx.Module
        Base distribution with sample(shape) and
        log_prob(x) methods
    solver : diffrax solver instance
        ODE solver (default: Dopri5())
    ode_kwargs : dict
        ODE solver options (rtol, atol)
    """

    vf_network: TokenisedVectorField
    base_dist: BaseDistribution

    def __init__(
        self,
        vf_network: TokenisedVectorField,
        base_dist: BaseDistribution,
        solver=None,
        ode_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize TFMPE.

        Parameters
        ----------
        vf_network : TokenisedVectorField
            Vector field network f(context, params, t) -> Array.
            Must be an nnx.Module so its state is captured and
            RNG streams are properly managed during training.
        base_dist : BaseDistribution
            Base distribution with sample(shape) and
            log_prob(x) methods. Example: NormalDistribution(rngs)
        solver : diffrax solver, optional
            ODE solver instance. Default: Heun()
        ode_kwargs : dict, optional
            ODE solver options with keys:
            - 'rtol': relative tolerance (default: 1e-5)
            - 'atol': absolute tolerance (default: 1e-5)
        """
        self.vf_network = vf_network
        self.base_dist = base_dist
        self.solver = solver if solver is not None else (
            diffrax.Dopri5()
        )

        # Set ODE solver parameters
        if ode_kwargs is None:
            ode_kwargs = {}
        self.ode_kwargs = {
            "rtol": ode_kwargs.get("rtol", 1e-5),
            "atol": ode_kwargs.get("atol", 1e-5),
        }

        # Validate ODE parameters
        if self.ode_kwargs["rtol"] <= 0 or (
            self.ode_kwargs["atol"] <= 0
        ):
            raise ValueError(
                "ODE tolerances must be positive. "
                f"Got rtol={self.ode_kwargs['rtol']}, "
                f"atol={self.ode_kwargs['atol']}"
            )

    def sample_posterior_batched(
        self,
        tokens: Tokens,
        batch_size: int
    ) -> Tokens:
        samples = []
        target = tokens.sample_shape[0]
        def process_batch(i) -> Tokens:
            batch = jax.tree.map(
                lambda leaf: leaf[i:i+batch_size],
                tokens
            )
            return self.sample_posterior(batch)

        samples = [
            process_batch(i)
            for i
            in range(0, target, batch_size)
        ]

        return dataclasses.replace(
            tokens,
            data=jnp.concatenate([s.data for s in samples])
        )

    def sample_posterior(
        self,
        tokens: Tokens
    ) -> Tokens:
        """Generate posterior samples via forward ODE solving.

        Samples from base distribution into params.data and solves
        forward ODE from t=0 to t=1 using the vector field network.

        Parameters
        ----------
        tokens: Tokens
            parameter and observation tokens

        Returns
        -------
        Tokens
            Posterior samples with same structure metadata as params
        """
        # Sample from base distribution into params.data
        target_n_tokens = tokens.data.shape[tokens.sample_ndims] - tokens.partition_idx
        target_shape = tokens.sample_shape + (target_n_tokens, tokens.data.shape[-1])
        source_data = tokens.data.at[:, tokens.partition_idx:].set(
            self.base_dist.sample(target_shape)
        )
        source_samples = dataclasses.replace(
            tokens,
            data=source_data
        )

        vf_fn = _make_stateless(self.vf_network)

        param_axes = Tokens(
            data=0, #type: ignore
            labels=0,#type: ignore
            position=0,#type: ignore
            condition=0,#type: ignore
            partition_idx=source_samples.partition_idx,
            padding_mask=None if source_samples.padding_mask is None else 0,#type: ignore
            functional_inputs=None if source_samples.functional_inputs is None else 0,#type: ignore
            group_id=0,#type: ignore
        )

        def solve_params(tokens):
            # Solve forward ODE using ODE helper
            output_tokens = solve_forward_ode(
                vf_fn=vf_fn,
                tokens=tokens,
                solver=self.solver,
                rtol=self.ode_kwargs["rtol"],
                atol=self.ode_kwargs["atol"],
            )
            return output_tokens


        return jax.vmap(
            solve_params,
            in_axes=[param_axes]
        )(source_samples)

    def log_prob_posterior_samples(
        self,
        tokens: Tokens,
        n_epsilon: int = 10,
    ) -> Array:
        """Compute log probabilities for posterior samples.

        Uses FFJORD algorithm with augmented backward ODE solving
        and stochastic trace estimation.

        Parameters
        ----------
        theta : Tokens
            Posterior samples to evaluate
        context : Tokens
            Context tokens
        n_epsilon : int, optional
            Number of Hutchinson trace samples (default: 10)

        Returns
        -------
        Array
            Log probability scalar
        """
        # Solve augmented ODE to get trace-based log determinant
        rng = jax.random.PRNGKey(0)

        def solve_log_prob(tokens):
            theta_tokens, log_det = solve_augmented_ode(
                vf_fn=_make_stateless(self.vf_network),
                tokens=tokens,
                solver=self.solver,
                rng=rng,
                rtol=self.ode_kwargs["rtol"],
                atol=self.ode_kwargs["atol"],
                n_epsilon=n_epsilon,
            )
            return theta_tokens, log_det

        theta_tokens, log_det = jax.vmap(
            solve_log_prob,
        )(tokens)

        # Compute log probability of base distribution
        log_prob_base = self.base_dist.log_prob(
            theta_tokens.data[:, theta_tokens.partition_idx:]
        )

        # Total log probability
        log_prob = log_prob_base + log_det

        return log_prob

def _make_stateless(model: TokenisedVectorField) -> TokenisedVectorField:
    if isinstance(model, nnx.Module):
        graphdef, state = nnx.split(model)
        state_flat, state_treedef = jax.tree.flatten(state)

        def stateless_vf_fn(
            tokens: Tokens, time: Array
        ) -> Array:
            rebuilt_state = state_treedef.unflatten(state_flat)
            model = nnx.merge(graphdef, rebuilt_state)
            model.eval()
            vf = model(tokens, time)
            return vf

        return stateless_vf_fn

    return model
