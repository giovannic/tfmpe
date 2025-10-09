import pytest
from jax import numpy as jnp, random as jr

from tfmpe.metrics.lc2st import run_lc2st

pytestmark = pytest.mark.slow
@pytest.mark.parametrize(
    "dim,train_size",
    [
        (1, 1_000),
        (10, 1_000),
        (100, 1_000),
    ]
)
def test_lc2st_on_theoretical_distribution(dim, train_size):
    key = jr.PRNGKey(0)
    # Create training data
    n_obs = 10

    sigma = 1e-1

    # Create calibration data (x, theta, theta_q)
    key_theta, key_x, key = jr.split(key, 3)
    theta_cal = jr.normal(key_theta, (train_size, dim))
    x_cal = jnp.repeat(theta_cal, n_obs, axis=1) + \
            jr.normal(key_x, (train_size, dim * n_obs)) * sigma

    # Compute theoretical posterior parameters
    # Prior: p(θ) = N(0, 1), Likelihood: p(y_i|θ) = N(θ, σ²) for n_obs observations
    # Posterior: p(θ|y) = N(μ_n, τ_n²) where τ_n² = σ²/(σ² + n), μ_n = Σy_i/(σ² + n)
    sigma_sq = sigma ** 2
    posterior_var = sigma_sq / (sigma_sq + n_obs)
    posterior_std = jnp.sqrt(posterior_var)

    # Posterior mean for calibration: sum of observations / (σ² + n)
    x_cal_reshaped = x_cal.reshape(train_size, dim, n_obs)
    posterior_mean_cal = x_cal_reshaped.sum(axis=2) / (sigma_sq + n_obs)

    # Good: sample from theoretical posterior
    key_post_cal, key = jr.split(key)
    theta_q_good = posterior_mean_cal + jr.normal(key_post_cal, (train_size, dim)) * posterior_std

    # Bad: incorrect sampling (just adds noise to prior samples, ignores observation weighting)
    key_bad_cal, key = jr.split(key)
    theta_q_bad = theta_cal + jr.normal(key_bad_cal, (train_size, dim)) * sigma + 1

    # Create calibration dataset (x, theta, theta_q)
    d_cal_good = (x_cal, theta_cal, theta_q_good)
    d_cal_bad = (x_cal, theta_cal, theta_q_bad)

    ev_key, key = jr.split(key)
    theta_truth = jr.normal(ev_key, (dim,))
    obs_key, key = jr.split(key)
    observation = jnp.repeat(theta_truth, n_obs) + jr.normal(obs_key, (dim * n_obs,)) * sigma

    # Posterior mean for the observation
    obs_reshaped = observation.reshape(dim, n_obs)
    posterior_mean_obs = obs_reshaped.sum(axis=1) / (sigma_sq + n_obs)

    # Good: sample from theoretical posterior
    key_post_obs, key = jr.split(key)
    theta_post_good = posterior_mean_obs + \
            jr.normal(key_post_obs, (train_size, dim)) * posterior_std

    # Bad: incorrect sampling (just adds noise to truth, ignores observation weighting)
    key_bad_obs, key = jr.split(key)
    theta_post_bad = theta_truth + jr.normal(key_bad_obs, (train_size, dim)) * sigma + 1

    r_good = run_lc2st(ev_key, d_cal_good, observation, theta_post_good)
    r_bad = run_lc2st(ev_key, d_cal_bad, observation, theta_post_bad)

    assert r_good.main_stat < r_bad.main_stat
    # assert r_good.main_stat < r_good.critical_value(0.95)
    # assert r_bad.main_stat > r_bad.critical_value(0.95)
    assert r_good.critical_value(.95) < 0.15
    assert r_bad.critical_value(.95) < 0.15
