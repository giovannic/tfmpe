import pytest
from jax import numpy as jnp, random as jr, tree
import flax.nnx as nnx
from tfmpe.metrics.lc2st import (
    _train_lc2st_classifiers,
    _evaluate_lc2st
)

from tfmpe.nn.classifier import (
    BinaryMLPClassifier,
    MultiBinaryMLPClassifier,
    MultiFoldBinaryMLPClassifier,
)

@pytest.mark.parametrize(
    "x_dim,theta_dim,n_layers,batch_size,latent_dim",
    [
        (8, 4, 1, 4, 16),         # small batch
        (16, 8, 2, 8, 16),        # larger model and batch
    ],
)
def test_binary_mlp_posterior_space_shape(x_dim, theta_dim, n_layers, batch_size, latent_dim):
    """
    Test that BinaryMLPClassifier __call__ accepts (x, theta) concatenated inputs
    and returns an array of shape matching the input batch dimensions.
    """
    # Initialize RNG and model
    key = jr.PRNGKey(0)
    model = BinaryMLPClassifier(
        dim=x_dim + theta_dim,  # Concatenated (x, theta) dimension
        n_layers=n_layers,
        activation=nnx.relu,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key),
    )

    # Generate test inputs
    key_x, key_theta = jr.split(key)
    x_shape = (batch_size, x_dim)
    theta_shape = (batch_size, theta_dim)
    x = jr.normal(key_x, x_shape)
    theta = jr.normal(key_theta, theta_shape)

    # Forward pass with u = concatenate([x, theta])
    u = jnp.concatenate([x, theta], axis=-1)
    out = model(u)

    # Assertions
    assert isinstance(out, jnp.ndarray), "Output must be a JAX array"
    # Expect one output per sample: output.shape == input.shape[:-1]
    assert out.shape == x.shape[:-1], (
        f"Expected output shape {x.shape[:-1]}, got {out.shape}"
    )
    # Check output dtype is floating-point
    assert jnp.issubdtype(out.dtype, jnp.floating), (
        f"Expected floating dtype, got {out.dtype}"
    )

@pytest.mark.parametrize(
    "x_dim,theta_dim,d_size,batch_size,latent_dim,n_fold,n_ens",
    [
        (8, 4, 100, 10, 16, 1, 1),
        (16, 8, 200, 20, 16, 10, 3)
    ],
)
def test__train_lc2st_classifiers_updates_params(x_dim, theta_dim, d_size, batch_size, latent_dim, n_fold, n_ens):
    """
    Test that _train_lc2st_classifiers runs for 1 epoch and updates both main and null classifier parameters.
    Uses (x, theta) and (x, theta_q) pairs for Local-Classifier 2 Sample Test.
    """
    # Setup
    key = jr.PRNGKey(42)
    # Instantiate main classifier for (x, theta) concatenated input
    main_classifier = MultiFoldBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        n_fold=n_fold,
        n=n_ens,
        activation=nnx.relu,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key),
    )
    
    # Instantiate null classifier
    num_null_classifiers = 3
    null_classifier = MultiBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        activation=nnx.relu,
        n=num_null_classifiers,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(jr.split(key)[1]),
    )
    
    # Generate calibration data (x, theta, theta_q)
    key_x, key_theta, key_theta_q = jr.split(key, 3)
    x = jr.normal(key_x, (d_size, x_dim))
    theta = jr.normal(key_theta, (d_size, theta_dim))
    theta_q = jr.normal(key_theta_q, (d_size, theta_dim))
    d_cal = (x, theta, theta_q)

    # Snapshot initial parameters (deep copy since nnx.state returns live view)
    initial_main_params = tree.map(jnp.copy, nnx.state(main_classifier))
    initial_null_params = tree.map(jnp.copy, nnx.state(null_classifier))

    # Train for 1 epoch
    _train_lc2st_classifiers(
        rng_key=key,
        d_cal=d_cal,
        classifier=main_classifier,
        null_classifier=null_classifier,
        num_epochs=1,
        batch_size=batch_size
    )

    # Check that main classifier parameters have changed
    leaves_before = tree.leaves(initial_main_params)
    leaves_after = tree.leaves(nnx.state(main_classifier))
    main_params_changed = any(
        not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after)
    )
    assert main_params_changed, "Expected main classifier parameters to update after training"
    
    # Check that null classifier parameters have changed
    leaves_before = tree.leaves(initial_null_params)
    leaves_after = tree.leaves(nnx.state(null_classifier))
    null_params_changed = any(
        not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after)
    )
    assert null_params_changed, "Expected null classifier parameters to update after training"

@pytest.mark.parametrize(
    "x_dim,theta_dim,n,n_fold,Nv,latent_dim",
    [
        (8, 4, 3, 5, 10, 16),
        (16, 8, 5, 10, 20, 16),
    ],
)
def test_multi_fold_classifier_4d_input_shape(x_dim, theta_dim, n, n_fold, Nv, latent_dim):
    """
    Test that MultiBinaryMLPClassifier handles 4D input correctly.
    When input shape is (Nv, n_classifiers, n_folds, x_dim + theta_dim), output should be 
    (Nv, n_classifiers).
    """
    # Initialize classifier
    cls = MultiFoldBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        activation=nnx.relu,
        n=n,
        n_fold=n_fold,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0)
    )
    
    # Create 3D input: (batch_dim, n_classifiers, n_folds, x_dim + theta_dim)
    key = jr.PRNGKey(42)
    u_eval_3d = jr.normal(key, shape=(Nv, n, n_fold, x_dim + theta_dim))
    
    # Forward pass
    prob = cls(u_eval_3d)
    
    # Verify output shape is (Nv, n_classifiers)
    expected_shape = (Nv, n, n_fold)
    assert prob.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {prob.shape}. "
        f"Input shape was {u_eval_3d.shape}"
    )
    
    # Verify output values are probabilities (between 0 and 1)
    assert jnp.all(prob >= 0) and jnp.all(prob <= 1), (
        "Output should be probabilities between 0 and 1"
    )


@pytest.mark.parametrize(
    "x_dim,theta_dim,n,Nv,latent_dim",
    [
        (8, 4, 3, 10, 16),
        (16, 8, 5, 20, 16),
    ],
)
def test_multi_classifier_3d_input_shape(x_dim, theta_dim, n, Nv, latent_dim):
    """
    Test that MultiBinaryMLPClassifier handles 3D input correctly.
    When input shape is (Nv, n_classifiers, x_dim + theta_dim), output should be 
    (Nv, n_classifiers).
    """
    # Initialize classifier
    cls = MultiBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        activation=nnx.relu,
        n=n,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0)
    )
    
    # Create 3D input: (batch_dim, n_classifiers, x_dim + theta_dim)
    key = jr.PRNGKey(42)
    u_eval_3d = jr.normal(key, shape=(Nv, n, x_dim + theta_dim))
    
    # Forward pass
    prob = cls(u_eval_3d)
    
    # Verify output shape is (Nv, n_classifiers)
    expected_shape = (Nv, n)
    assert prob.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {prob.shape}. "
        f"Input shape was {u_eval_3d.shape}"
    )
    
    # Verify output values are probabilities (between 0 and 1)
    assert jnp.all(prob >= 0) and jnp.all(prob <= 1), (
        "Output should be probabilities between 0 and 1"
    )

@pytest.mark.parametrize(
    "x_dim,theta_dim,Nv,num_null,latent_dim",
    [
        (8, 4, 10, 5, 16),
        (16, 8, 20, 10, 16),
    ],
)
def test__evaluate_lc2st_output(x_dim, theta_dim, Nv, num_null, latent_dim):
    """
    Test _evaluate_lc2st returns proper-shaped, positive statistics for untrained classifiers.
    Uses (x, theta) concatenated inputs for Local-Classifier 2 Sample Test evaluation.
    """
    key = jr.PRNGKey(123)
    
    # Main classifier for (x, theta) concatenated input
    key_main = jr.PRNGKey(1)
    main_clf = MultiFoldBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        activation=nnx.relu,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(key_main),
        n=2,
        n_fold=2
    )
    
    # Null classifiers for (x, theta) concatenated input
    null_clf = MultiBinaryMLPClassifier(
        dim=x_dim + theta_dim,
        n_layers=2,
        activation=nnx.relu,
        n=num_null,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0),
    )
    
    # Generate observation and posterior samples
    key_obs, key_post = jr.split(key)
    observation = jr.normal(key_obs, (x_dim,))
    posterior_samples = jr.normal(key_post, (Nv, theta_dim))
    
    # Evaluate
    null_stats, t_stat = _evaluate_lc2st(
        observation=observation,
        posterior_samples=posterior_samples,
        main_classifier=main_clf,
        null_classifier=null_clf,
    )

    # Assertions
    assert null_stats.shape == (num_null,), f"Expected shape ({num_null},), got {null_stats.shape}"
    
    assert t_stat.shape == (), (
        f"Expected shape (), got {t_stat.shape}"
    )
    assert t_stat >= 0, "Expected non-negative values"
