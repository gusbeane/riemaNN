"""Training primitives: optimizer factory, train step, loop, checkpoint I/O.

Pure building blocks with no dependency on the Experiment class.
Experiment orchestration (skip-if-checkpoint, directory management) lives
in experiment.py; these functions remain importable from notebooks or
ad-hoc scripts without dragging in that layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state as flax_train_state
from tqdm import tqdm

from .physics import GAS_STATE_DIM


# --- optimizer factory --------------------------------------------------------


def build_optimizer(spec: dict[str, Any]) -> optax.GradientTransformation:
    """Build an optax optimizer from a spec dict.

    Supported types: adam, adamw, sgd. Additional keys are passed through
    to the optax constructor. Raises on unknown type.
    """
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "adam":
        return optax.adam(**spec)
    if kind == "adamw":
        return optax.adamw(**spec)
    if kind == "sgd":
        return optax.sgd(**spec)
    raise ValueError(f"Unknown optimizer type: {kind!r}")


# --- train state --------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    batch_size_hint: int = 256,
) -> flax_train_state.TrainState:
    """Initialize params with a dummy batch of the standard gas-state shape."""
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


# --- train step and loop ------------------------------------------------------


def make_train_step(loss_fn: Callable) -> Callable:
    """Wrap `loss_fn(params, apply_fn, gas_states_log) -> (loss, metrics)`
    into a jitted `(state, gas_states_log) -> (state, loss, metrics)`.
    """

    @jax.jit
    def train_step(state, gas_states_log):
        def _wrapped(params):
            return loss_fn(params, state.apply_fn, gas_states_log)

        (loss, metrics), grads = jax.value_and_grad(_wrapped, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics

    return train_step


def run_training_loop(
    state: flax_train_state.TrainState,
    train_step: Callable,
    sampler: Callable,
    rng: jax.Array,
    n_epochs: int,
    batch_size: int,
    *,
    log_every: int = 2000,
    desc: str = "train",
) -> tuple[flax_train_state.TrainState, jnp.ndarray]:
    """Simple training loop; accumulates per-step loss and returns (state, loss_trace)."""
    loss_trace: list[float] = []
    pbar = tqdm(range(n_epochs), desc=desc)
    for epoch in pbar:
        rng, batch_key = jr.split(rng)
        batch = sampler(batch_key, batch_size)
        state, loss, _metrics = train_step(state, batch)
        loss_trace.append(float(loss))
        if epoch % log_every == 0:
            pbar.set_postfix(loss=f"{loss:.2e}")
    return state, jnp.array(loss_trace)


# --- checkpoint I/O -----------------------------------------------------------


def save_checkpoint(path: Path, state: flax_train_state.TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(state))


def load_checkpoint(
    path: Path, template_state: flax_train_state.TrainState
) -> flax_train_state.TrainState:
    return from_bytes(template_state, path.read_bytes())


def save_loss_trace(path: Path, loss_trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(loss_trace))


def load_loss_trace(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)
