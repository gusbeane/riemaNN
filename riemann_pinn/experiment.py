"""Experiment DSL: compose training runs from a list of Phases.

An Experiment bundles (model, domain, seed, phases). Each Phase owns its
optimizer, loss, step kind, and duration. The runner executes phases
sequentially, carrying params across phase boundaries while rebuilding
opt_state from scratch for each new optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from .train import (
    R2_quasirandom,
    create_train_state,
    make_lbfgs_train_step,
    make_train_step,
    residual_loss,
    residual_loss_supervised,
    residual_loss_normalized,
    run_training_loop,
    uniform_log,
)


LOSS_FNS: dict[str, Callable] = {
    "fstar": residual_loss,
    "pstar": residual_loss_supervised,
    "fstar_normalized": residual_loss_normalized,
}

STEP_BUILDERS: dict[str, Callable] = {
    "sgd": make_train_step,
    "lbfgs": make_lbfgs_train_step,
}

SAMPLERS: dict[str, Callable] = {
    "uniform": uniform_log,
    "r2": R2_quasirandom,
}


@dataclass
class Phase:
    """One training phase with a fresh optimizer.

    `sampler` is a key into SAMPLERS; the runner calls it with the
    Experiment's domain. `split_key_every=1` resamples batches every
    step; a value >= n_epochs keeps a single fixed batch for the whole
    phase (typical for L-BFGS).
    """

    n_epochs: int
    batch_size: int
    optimizer: optax.GradientTransformation
    loss: str = "fstar"
    step_kind: str = "sgd"
    name: str = "phase"
    log_every: int = 100
    split_key_every: int = 1
    sampler: str = "uniform"


@dataclass
class Experiment:
    model: nn.Module
    domain: dict
    phases: list[Phase]
    seed: int = 42
    corner_every: int = 100


# --- phase factories ----------------------------------------------------------


def adam_cosine(
    *,
    n_epochs: int,
    lr: float = 1e-3,
    alpha: float = None,
    batch_size: int = 256,
    loss: str = "fstar",
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    sampler: str = "uniform",
    log_every: int = 100,
    name: str = "adam_cosine",
) -> Phase:
    """AdamW with cosine-decayed LR and global-norm clipping."""
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs, alpha=alpha)
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    return Phase(
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=tx,
        loss=loss,
        step_kind="sgd",
        name=name,
        log_every=log_every,
        sampler=sampler,
    )


def adam(
    *,
    n_epochs: int,
    lr: float = 1e-3,
    batch_size: int = 256,
    loss: str = "fstar",
    sampler: str = "uniform",
    log_every: int = 100,
    name: str = "adam",
) -> Phase:
    """Plain Adam at constant LR."""
    tx = optax.adam(learning_rate=lr)
    return Phase(
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=tx,
        loss=loss,
        step_kind="sgd",
        name=name,
        log_every=log_every,
        sampler=sampler,
    )


def lbfgs(
    *,
    n_epochs: int,
    batch_size: int = 256,
    loss: str = "fstar",
    learning_rate: float = 1.0,
    memory_size: int = 10,
    split_key_every: Optional[int] = None,
    sampler: str = "uniform",
    log_every: int = 100,
    name: str = "lbfgs",
) -> Phase:
    """L-BFGS with a fixed batch by default (split_key_every=n_epochs+1)."""
    tx = optax.lbfgs(learning_rate=learning_rate, memory_size=memory_size)
    if split_key_every is None:
        split_key_every = n_epochs + 1
    return Phase(
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=tx,
        loss=loss,
        step_kind="lbfgs",
        name=name,
        log_every=log_every,
        split_key_every=split_key_every,
        sampler=sampler,
    )


# --- runner -------------------------------------------------------------------


def run_experiment(exp: Experiment, *, corner_callback: Optional[Callable] = None):
    """Execute all phases. Returns (final_state, full_loss_trace, per_phase_traces).

    If ``corner_callback`` is given, it is invoked as ``corner_callback(state, step)``
    every ``exp.corner_every`` global training steps. The step counter persists
    across phase boundaries.
    """
    rng = jr.PRNGKey(exp.seed)

    state = None
    traces: list[jnp.ndarray] = []
    n_phases = len(exp.phases)
    global_step_offset = 0
    for i, phase in enumerate(exp.phases):
        if state is None:
            state = create_train_state(
                rng, exp.model, phase.optimizer, batch_size_hint=phase.batch_size,
            )
        else:
            state = TrainState.create(
                apply_fn=state.apply_fn, params=state.params, tx=phase.optimizer,
            )

        loss_fn = LOSS_FNS[phase.loss]
        step = STEP_BUILDERS[phase.step_kind](loss_fn)
        sampler_fn = SAMPLERS[phase.sampler]
        def sampler(key, batch_size, _fn=sampler_fn):
            return _fn(key, batch_size, **exp.domain)

        if corner_callback is not None:
            every = exp.corner_every
            def step_cb(s, global_step, _cb=corner_callback, _every=every):
                if global_step % _every == 0:
                    _cb(s, global_step)
        else:
            step_cb = None

        phase_rng = jr.fold_in(rng, i + 1)
        desc = f"[{i + 1}/{n_phases}] {phase.name}"
        state, trace = run_training_loop(
            state, step, sampler, phase_rng,
            n_epochs=phase.n_epochs, batch_size=phase.batch_size,
            desc=desc, log_every=phase.log_every,
            split_key_every=phase.split_key_every,
            step_callback=step_cb, step_offset=global_step_offset,
        )
        traces.append(trace)
        global_step_offset += phase.n_epochs

    full_trace = jnp.concatenate(traces) if traces else jnp.array([])
    return state, full_trace, traces


def build_template_state(exp: Experiment) -> TrainState:
    """Template TrainState for deserializing a checkpoint of this experiment.

    Checkpoints are saved after the final phase, so the template must use
    its optimizer (opt_state shape must match what was serialized).
    """
    rng = jr.PRNGKey(exp.seed)
    last = exp.phases[-1]
    return create_train_state(
        rng, exp.model, last.optimizer, batch_size_hint=last.batch_size,
    )
