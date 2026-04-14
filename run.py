"""Train a PINN for Riemann star pressure: Adam (10k) -> L-BFGS (1k), w256 d3."""

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state as flax_train_state

from riemann_pinn.model import StarPressureMLP
from riemann_pinn.train import (
    build_optimizer, create_train_state, evaluate_holdout,
    load_checkpoint, load_loss_trace, make_lbfgs_train_step,
    make_train_step, residual_loss, run_training_loop, save_checkpoint,
    save_loss_trace, uniform_log,
)
from riemann_pinn.plot import plot_loss, plot_slice

# --- config -------------------------------------------------------------------

NAME = "al_w256_d3"
MODEL = StarPressureMLP(width=256, depth=3)
SEED = 42
ADAM_EPOCHS = 10_000
ADAM_LR = 1e-3
ADAM_BATCH = 256
LBFGS_EPOCHS = 1_000
LBFGS_BATCH = 4096
LBFGS_MEMORY = 10

# --- paths --------------------------------------------------------------------

OUT_DIR = Path("outputs") / NAME
CKPT_PATH = OUT_DIR / "checkpoint.msgpack"
LOSS_PATH = OUT_DIR / "loss.npy"
METRICS_PATH = OUT_DIR / "metrics.json"
LOSS_PLOT = OUT_DIR / "plots" / "loss.png"
SLICE_PLOT = OUT_DIR / "plots" / "slice.png"


def train_model():
    rng = jr.PRNGKey(SEED)

    # Phase 1: Adam
    adam_opt = build_optimizer({"type": "adam", "learning_rate": ADAM_LR})
    state = create_train_state(rng, MODEL, adam_opt, batch_size_hint=ADAM_BATCH)
    adam_step = make_train_step(residual_loss)
    state, trace_adam = run_training_loop(
        state, adam_step, uniform_log, jr.fold_in(rng, 1),
        n_epochs=ADAM_EPOCHS, batch_size=ADAM_BATCH,
        desc=f"{NAME} adam", log_every=2000,
    )

    # Phase 2: L-BFGS
    lbfgs_opt = build_optimizer({"type": "lbfgs", "memory_size": LBFGS_MEMORY})
    state = flax_train_state.TrainState.create(
        apply_fn=state.apply_fn, params=state.params, tx=lbfgs_opt,
    )
    lbfgs_step = make_lbfgs_train_step(residual_loss)
    state, trace_lbfgs = run_training_loop(
        state, lbfgs_step, uniform_log, jr.fold_in(rng, 2),
        n_epochs=LBFGS_EPOCHS, batch_size=LBFGS_BATCH,
        desc=f"{NAME} lbfgs", log_every=100,
    )

    return state, jnp.concatenate([trace_adam, trace_lbfgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    training_time_s = None
    if CKPT_PATH.is_file() and not args.retrain:
        print(f"Loading checkpoint from {CKPT_PATH}")
        rng = jr.PRNGKey(SEED)
        template = create_train_state(
            rng, MODEL,
            build_optimizer({"type": "lbfgs", "memory_size": LBFGS_MEMORY}),
        )
        state = load_checkpoint(CKPT_PATH, template)
        loss_trace = load_loss_trace(LOSS_PATH)
    else:
        t0 = time.monotonic()
        state, loss_trace = train_model()
        training_time_s = round(time.monotonic() - t0, 1)
        save_checkpoint(CKPT_PATH, state)
        save_loss_trace(LOSS_PATH, loss_trace)

    # Evaluate
    metrics = evaluate_holdout(state)
    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Plot
    if not args.skip_plots:
        if loss_trace is not None:
            plot_loss(loss_trace, LOSS_PLOT, title=f"Training loss — {NAME}")
        plot_slice(state, SLICE_PLOT)


if __name__ == "__main__":
    main()
