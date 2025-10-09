from jax import numpy as jnp, random as jr, tree
from flax import nnx
from jaxtyping import PyTree, Array
from typing import Tuple, Callable, List, Optional, Any, TypeVar
from tqdm import tqdm

M = TypeVar('M', bound=nnx.Module)

def fit_nn(
    model: M,
    train: PyTree,
    val: Optional[PyTree],
    opt: nnx.Optimizer,
    loss: Callable[[Any, PyTree, Array], Array],
    n_iter: int,
    batch_size: int,
    rng: Array,
    delta: float = 0.0,
    patience: int = 0,
) -> Tuple[M, Tuple[Array, Array]]:
    """Train a model with mini-batch SGD and optional early stopping.

    Parameters
    ----------
    model : nnx.Module
        Model to train
    train : PyTree
        Training data
    val : PyTree, optional
        Validation data. Required if patience > 0.
    opt : nnx.Optimizer
        NNX optimizer instance already initialized with model
    loss : Callable
        Loss function (model, batch, rng) -> scalar loss
    n_iter : int
        Maximum number of training iterations (epochs)
    batch_size : int
        Number of samples per batch
    rng : PRNGKeyArray
        Random number generator key
    delta : float, optional
        Minimum improvement in validation loss to reset patience counter.
        Default is 0.0 (any improvement counts).
    patience : int, optional
        Number of epochs to wait for improvement before stopping.
        Set to 0 to disable early stopping. Default is 0.

    Returns
    -------
    Tuple[M, Tuple[Array, Array]]
        Trained model (with best weights if early stopped) and tuple of:
        - training losses shape (n_epochs,) where n_epochs <= n_iter
        - validation losses shape (n_epochs,) where n_epochs <= n_iter
    """
    n_train = tree.leaves(train)[0].shape[0]
    n_batches = n_train // batch_size

    # JIT-compiled training step
    @nnx.jit
    def train_step(
        model: nnx.Module,
        opt_model: nnx.Optimizer,
        batch: PyTree,
        rng_key: Array,
    ) -> Array:
        def model_loss(model: nnx.Module) -> Array:
            return loss(
                model,
                batch,
                rng_key,
            )

        batch_loss, grads = nnx.value_and_grad(model_loss)(model)
        opt_model.update(model, grads)
        return batch_loss

    # JIT-compiled validation loss
    @nnx.jit
    def compute_val_loss(
        model: nnx.Module,
        val: PyTree,
        rng: Array,
    ) -> Array:
        return loss(
            model,
            val,
            rng
        )

    # Pre-split RNG keys for all epochs
    epoch_rngs = jr.split(rng, n_iter)

    # Accumulate losses in Python lists
    train_losses_list: List[Array] = []
    val_losses_list: List[Array] = []

    # Early stopping state
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None
    early_stopping_enabled = patience > 0 and val is not None

    # Python loop over epochs with progress bar
    pbar = tqdm(range(n_iter), desc="Training")
    for epoch in pbar:
        epoch_rng = epoch_rngs[epoch]

        # Shuffle training data
        epoch_rng, perm_key = jr.split(epoch_rng)
        perm = jr.permutation(perm_key, n_train)

        # Generate RNG keys for each batch
        epoch_rng, batch_rng = jr.split(epoch_rng)
        batch_rngs = jr.split(batch_rng, n_batches)

        # Accumulate batch losses for this epoch
        batch_losses: List[Array] = []

        # Python loop over batches
        for batch_idx in range(n_batches):
            # Extract batch data
            start = batch_idx * batch_size
            end = start + batch_size
            batch = tree.map(
                lambda x: x[perm[start:end]],
                train,
            )

            # Run JIT-compiled training step
            batch_loss = train_step(
                model,
                opt,
                batch,
                batch_rngs[batch_idx]
            )
            batch_losses.append(batch_loss)

        # Average training loss across batches
        train_loss = jnp.mean(jnp.stack(batch_losses))
        train_losses_list.append(train_loss)

        # Compute validation loss
        if val is not None:
            epoch_rng, val_key = jr.split(epoch_rng)

            val_loss = compute_val_loss(model, val, val_key)
            val_losses_list.append(val_loss)

            # Update progress bar with losses
            pbar.set_postfix(train_loss=f"{float(train_loss):.4f}", val_loss=f"{float(val_loss):.4f}")

            # Early stopping check
            if early_stopping_enabled:
                if best_val_loss - float(val_loss) > delta:
                    # Improvement found
                    best_val_loss = float(val_loss)
                    epochs_without_improvement = 0
                    # Save best model state (deep copy since nnx.state returns live view)
                    best_state = tree.map(jnp.copy, nnx.state(model))
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        # Restore best model and stop
                        assert best_state is not None
                        nnx.update(model, best_state)
                        break
        else:
            pbar.set_postfix(train_loss=f"{float(train_loss):.4f}")

    # Stack losses into arrays (may be shorter than n_iter if early stopped)
    train_losses = jnp.stack(train_losses_list)
    if val is not None:
        val_losses = jnp.stack(val_losses_list)
    else:
        val_losses = jnp.array([])

    return model, (train_losses, val_losses)
