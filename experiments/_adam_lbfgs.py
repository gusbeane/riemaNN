"""Shared two-phase train function: Adam warm-start then L-BFGS polish.

Import this in experiment scripts rather than duplicating the logic.
"""

import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state as flax_train_state

from riemann_pinn import losses, training

# Defaults for the two phases — experiments can override via exp fields.
ADAM_EPOCHS = 5_000
ADAM_BATCH = 256
ADAM_LR = 1e-3
LBFGS_EPOCHS = 500
LBFGS_BATCH = 4096
LBFGS_MEMORY = 10


def adam_then_lbfgs(
    exp,
    *,
    adam_epochs=ADAM_EPOCHS,
    adam_batch=ADAM_BATCH,
    adam_lr=ADAM_LR,
    lbfgs_epochs=LBFGS_EPOCHS,
    lbfgs_batch=LBFGS_BATCH,
    lbfgs_memory=LBFGS_MEMORY,
):
    """Two-phase training: Adam warm-start, then L-BFGS polish."""
    loss_fn = losses.make_loss_fn(exp.target, exp.loss_impl, **exp.loss_kwargs)
    rng = jr.PRNGKey(exp.seed)

    # Phase 1: Adam
    adam_opt = training.build_optimizer({"type": "adam", "learning_rate": adam_lr})
    state = training.create_train_state(rng, exp.model, adam_opt, batch_size_hint=adam_batch)
    adam_step = training.make_train_step(loss_fn)
    state, trace_adam = training.run_training_loop(
        state, adam_step, exp.sampler,
        jr.fold_in(rng, 1),
        n_epochs=adam_epochs,
        batch_size=adam_batch,
        desc=f"{exp.name} adam",
        log_every=2000,
    )

    # Phase 2: L-BFGS — must create fresh optimizer state (curvature history)
    lbfgs_opt = training.build_optimizer({"type": "lbfgs", "memory_size": lbfgs_memory})
    state = flax_train_state.TrainState.create(
        apply_fn=state.apply_fn, params=state.params, tx=lbfgs_opt,
    )
    lbfgs_step = training.make_lbfgs_train_step(loss_fn)
    state, trace_lbfgs = training.run_training_loop(
        state, lbfgs_step, exp.sampler,
        jr.fold_in(rng, 2),
        n_epochs=lbfgs_epochs,
        batch_size=lbfgs_batch,
        desc=f"{exp.name} lbfgs",
        log_every=100,
    )

    return state, jnp.concatenate([trace_adam, trace_lbfgs])
