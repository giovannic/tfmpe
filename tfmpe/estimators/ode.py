"""ODE solving helpers for flow matching estimators."""

from typing import Callable, Tuple
import dataclasses

import diffrax
import jax
import jax.numpy as jnp
from jax import tree
from jaxtyping import Array, Scalar

from ..preprocessing.tokens import Tokens


def _make_step_controller(solver, rtol, atol):
    """Pick step controller based on solver capabilities.

    Adaptive solvers (e.g. Dopri5, Heun) use PIDController.
    Fixed-order solvers without error estimates (e.g. Euler)
    use ConstantStepSize.
    """
    if isinstance(solver, diffrax.AbstractAdaptiveSolver):
        return diffrax.PIDController(rtol=rtol, atol=atol), None
    return diffrax.ConstantStepSize(), 0.01


def solve_forward_ode(
    vf_fn: Callable[[Tokens, Scalar], Array],
    tokens: Tokens,
    solver,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tokens:
    """Solve forward ODE from t=0 to t=1 for sampling.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(context, params, t) -> Array
    tokens: Tokens
        Singleton tokens for integration
    solver : diffrax solver
        Solver instance (e.g., Diffrax Dopri5)
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver

    Returns
    -------
    Tokens
        Final target tokens at t=1
    """
    def ode_func(t, y_data: Array, args) -> Array:
        data = tokens.data.at[tokens.partition_idx:].set(y_data)
        tokens_t = dataclasses.replace(
            tokens,
            data=data
        )
        tokens_t = tree.map(
            lambda leaf: leaf[None,...],
            tokens_t
        )
        v = vf_fn(tokens_t, t)
        return v[0]

    y0 = tokens.data[tokens.partition_idx:]

    step_controller, dt0 = _make_step_controller(
        solver, rtol, atol
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_func),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=dt0,
        y0=y0,
        stepsize_controller=step_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    assert solution.ys is not None
    output_tokens = dataclasses.replace(
        tokens,
        data=tokens.data.at[tokens.partition_idx:].set(solution.ys[0])
    )
    return output_tokens

def solve_augmented_ode(
    vf_fn: Callable[[Tokens, Scalar], Array],
    tokens: Tokens,
    solver,
    rng: Array,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    n_epsilon: int = 1,
) -> Tuple[Tokens, Scalar]:
    """Solve augmented backward ODE for FFJORD log probability.

    Augmented state: [params, log_det_jacobian]

    Solves backward from t=1 to t=0 using FFJORD algorithm.
    Uses stochastic trace estimation via VJP.

    Parameters
    ----------
    vf_fn : Callable
        Vector field function f(context, params, t) -> Array
    context : Tokens
        Context tokens (fixed during integration)
    params : Tokens
        Parameter tokens being evolved
    solver : diffrax solver
        Solver instance
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    rng : PRNGKeyArray, optional
        PRNG key for sampling epsilon. If None, uses PRNGKey(0)
    n_epsilon : int
        Number of trace samples for Hutchinson estimator

    Returns
    -------
    Tuple[Tokens, Scalar]
        (final_params at t=0, final_log_det_jacobian)
    """
    # Pre-sample epsilon array for stochastic trace estimation
    n_tokens = tokens.data.shape[0] - tokens.partition_idx
    epsilon_array = jax.random.normal(
        rng, (n_epsilon, n_tokens) + tokens.data.shape[1:]
    )

    # Backward integration: negate both time and vector field
    vector_sign = -1.0

    def augmented_ode_func(
        t,
        aug_state: Tuple[Array, Scalar],
        args,
    ) -> Tuple[Array, Scalar]:
        y_data, _ = aug_state

        # Map ODE time back to original time direction
        # ODE integrates from t=1 to t=0 as t goes from -1 to 0
        # So actual time is -t
        actual_time = -t

        # Compute vector field at actual time
        def vf_wrapper(y_inner):
            # Apply vector sign to both time and dynamics
            inner_data = tokens.data.at[tokens.partition_idx:].set(y_inner)
            inner_tokens = dataclasses.replace(
                tokens,
                data = inner_data
            )
            inner_tokens = tree.map(
                lambda leaf: leaf[None, ...],
                inner_tokens
            )
            return vector_sign * vf_fn(
                inner_tokens, actual_time
            )[0]

        # Compute trace via stochastic VJP
        # tr(∂f/∂x) ≈ mean_i[eps_i^T @ (∂f/∂x)^T @ eps_i]
        _, vjp_fn = jax.vjp(vf_wrapper, y_data)

        def compute_trace_single(eps):
            g = vjp_fn(eps)[0]
            return jnp.sum(g * eps)

        # Average over all epsilon samples
        trace_estimates = jax.vmap(
            compute_trace_single, in_axes=0
        )(epsilon_array)
        trace_estimate = jnp.mean(trace_estimates)

        # Return augmented dynamics
        f_y = vf_wrapper(y_data)
        return (f_y, -trace_estimate)

    # Initial augmented state at t=1
    aug_y0 = (
        tokens.data[tokens.partition_idx:],
        jnp.array(0.0)
    )

    step_controller, dt0 = _make_step_controller(
        solver, rtol, atol
    )
    # Integrate from -1 to 0 (equivalent to t=1 to t=0)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(augmented_ode_func),
        solver,
        t0=-1.0,
        t1=0.0,
        dt0=dt0,
        y0=aug_y0,
        stepsize_controller=step_controller,
        saveat=diffrax.SaveAt(t1=True),
    )

    # solution.ys is a tuple (y_trajectory, log_det_trajectory)
    # With saveat=SaveAt(t1=True), each has 1 element along time
    assert solution.ys is not None
    final_y_data = solution.ys[0][0]
    final_log_det = solution.ys[1][0]
    final_data = tokens.data.at[tokens.partition_idx:].set(final_y_data)
    final_tokens = dataclasses.replace(
        tokens,
        data = final_data
    )
    return final_tokens, final_log_det
