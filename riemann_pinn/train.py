"""Training primitives: Experiment / Stage / Phase, pipeline runner with
prev-stage threading, checkpoint I/O, holdout evaluation. All in 3D
delta-space."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state as flax_train_state
from tqdm import tqdm

from . import physics
from .data import DataSet, R2QuasirandomSampler, Sampler, UniformSampler  # noqa: F401
from .physics import GAS_STATE_DIM


# --- losses ------------------------------------------------------------------


def mse_loss(params, apply_fn, gas_states, targets):
    """Mean squared error between the network output and runner-derived targets."""
    pred = apply_fn({"params": params}, gas_states)
    return jnp.mean((pred - targets) ** 2)

def mse_log_loss(params, apply_fn, gas_states, targets):
    """Mean squared error between the network output and runner-derived targets."""
    pred = apply_fn({"params": params}, gas_states)
    return jnp.mean((jnp.log10(pred) - jnp.log10(targets)) ** 2)


# --- composition defaults ----------------------------------------------------


def _multiplicative_make_targets(prev_running, pstar_true):
    """Default: predict the multiplicative residual pstar_true / prev_running.

    For stage 0, prev_running is jnp.ones(...) so this returns pstar_true."""
    return pstar_true / prev_running


def _multiplicative_combine(prev_running, stage_output):
    """Default: running prediction is the running product of stage outputs."""
    return prev_running * stage_output


# --- experiment / stage / phase ----------------------------------------------


@dataclass
class Phase:
    """One training phase within a stage. A stage may have multiple phases
    (e.g., AdamW followed by a cosine-decayed AdamW)."""
    tx: optax.GradientTransformation
    n_epochs: int
    loss: Callable
    batch_size: int
    sampler: Sampler | DataSet
    log_every: int = 200
    name: str = "phase"


@dataclass
class Stage:
    """One stage of the pipeline. Trains a single NN against targets derived
    from the prior stages' running prediction.

    make_targets and combine default to multiplicative-residual composition.
    Stage 0 sees prev_running = jnp.ones(...), so the defaults reduce to
    supervised regression against pstar_true."""
    name: str
    model: nn.Module
    phases: list[Phase]
    make_targets: Callable = field(default=_multiplicative_make_targets)
    combine: Callable = field(default=_multiplicative_combine)


@dataclass
class Experiment:
    """One pipelined experiment: an ordered list of Stages.

    domain: sampling + evaluation region. Keys drho_range, dp_range, du_range.
    output_root: optional override for the parent directory. If unset, defaults
        to outputs/<file_stem>/.
    """
    name: str
    stages: list[Stage]
    domain: dict
    seed: int = 42
    output_root: str | Path | None = None


# --- train-state + step ------------------------------------------------------


def create_train_state(rng, model, tx, batch_size_hint: int = 256):
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _make_step(stage: Stage, prev_specs: list[tuple], loss_fn: Callable) -> Callable:
    """Build the JIT'd train step for `stage`, closing over the prior stages.

    prev_specs: list of (apply_fn, params, combine) for each prior stage.
    Each step:
      1. computes pstar_true from gas_states via vmap(find_pstar);
      2. evaluates each prior stage at gas_states and runs combine to get
         prev_running;
      3. derives targets = stage.make_targets(prev_running, pstar_true);
      4. computes loss_fn against those targets and applies one optimizer step.
    """
    make_targets = stage.make_targets

    @jax.jit
    def step(state, gas_states):
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
        prev_running = jnp.ones_like(pstar_true)
        for prev_apply, prev_params, prev_combine in prev_specs:
            out = prev_apply({"params": prev_params}, gas_states)
            prev_running = prev_combine(prev_running, out)
        targets = make_targets(prev_running, pstar_true)
        loss, grads = jax.value_and_grad(
            lambda p: loss_fn(p, state.apply_fn, gas_states, targets),
        )(state.params)
        return state.apply_gradients(grads=grads), loss

    return step


# --- runner ------------------------------------------------------------------


def _draw_gas_states(sampler, rng, batch_size):
    if isinstance(sampler, Sampler):
        return sampler.draw_batch(rng, batch_size)
    if isinstance(sampler, DataSet):
        gas_states, _ = sampler.draw_batch(batch_size)
        return gas_states
    raise ValueError(f"Unsupported sampler type: {type(sampler)}")


def run_stage(
    stage: Stage, prev_specs: list[tuple],
    *, exp_seed: int, stage_index: int,
):
    """Run all phases for one stage. Returns (state, full_loss_trace, per_phase_traces).

    All randomness — weight init and per-phase batch sampling — is derived
    deterministically from `(exp_seed, stage_index, phase_index)` via nested
    `jr.fold_in`. There is no separate `rng` argument."""
    stage_rng = jr.fold_in(jr.PRNGKey(exp_seed), stage_index)
    state = None
    traces: list[jnp.ndarray] = []
    for j, phase in enumerate(stage.phases):
        phase_rng = jr.fold_in(stage_rng, j)
        if state is None:
            init_rng, phase_rng = jr.split(phase_rng)
            state = create_train_state(init_rng, stage.model, phase.tx, batch_size_hint=phase.batch_size)
        else:
            state = flax_train_state.TrainState.create(
                apply_fn=state.apply_fn, params=state.params, tx=phase.tx,
            )
        step_fn = _make_step(stage, prev_specs, phase.loss)
        loss_trace: list[float] = []
        pbar = tqdm(range(phase.n_epochs), desc=f"  phase[{j}] {phase.name}")
        for epoch in pbar:
            phase_rng, batch_key = jr.split(phase_rng)
            gas_states = _draw_gas_states(phase.sampler, batch_key, phase.batch_size)
            state, loss = step_fn(state, gas_states)
            loss_trace.append(float(loss))
            if epoch % phase.log_every == 0:
                pbar.set_postfix(loss=f"{loss:.2e}")
        traces.append(jnp.array(loss_trace))
    full_trace = jnp.concatenate(traces) if traces else jnp.array([])
    return state, full_trace, traces


def build_template_state(stage: Stage):
    """Template TrainState for deserializing a checkpoint of `stage`. Uses
    the last phase's optimizer (matches save_checkpoint semantics)."""
    rng = jr.PRNGKey(0)
    last = stage.phases[-1]
    return create_train_state(rng, stage.model, last.tx, batch_size_hint=last.batch_size)


# --- pipeline prediction -----------------------------------------------------


def predict_pipeline(stage_states, gas_states):
    """Run the full pipeline forward.

    stage_states: list of (Stage, TrainState) in order.
    Returns the final running prediction (B,).
    """
    pred = jnp.ones((gas_states.shape[0],))
    for stage, state in stage_states:
        out = state.apply_fn({"params": state.params}, gas_states)
        pred = stage.combine(pred, out)
    return pred


# --- checkpoint I/O ----------------------------------------------------------


def save_checkpoint(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(state))


def load_checkpoint(path: Path, template_state):
    return from_bytes(template_state, path.read_bytes())


def save_loss_trace(path: Path, loss_trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(loss_trace))


def load_loss_trace(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)


# --- evaluation --------------------------------------------------------------


def evaluate_holdout(stage_states, n_samples: int = 20_000, seed: int = 999, **domain_kwargs):
    """Residual + pressure-error metrics on a uniform holdout batch.

    stage_states: list of (Stage, TrainState) in order. Pass a truncated
    list to evaluate only the first k stages."""
    rng = jr.PRNGKey(seed)
    sampler = UniformSampler(**domain_kwargs)
    gas_states = sampler.draw_batch(rng, n_samples)

    pstar_nn = predict_pipeline(stage_states, gas_states)

    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)

    metrics: dict[str, Any] = {}
    abs_f = jnp.abs(fstar_vals)
    metrics["median_abs_fstar"] = float(jnp.median(abs_f))
    metrics["p95_abs_fstar"]    = float(jnp.percentile(abs_f, 95.0))

    dlogp = jnp.log10(pstar_nn) - jnp.log10(pstar_true)
    abs_dlogp = np.asarray(jnp.abs(dlogp))
    metrics["median_abs_delta_log10_p"] = float(np.nanmedian(abs_dlogp))
    metrics["p95_abs_delta_log10_p"]    = float(np.nanpercentile(abs_dlogp, 95.0))

    abs_absolute = np.asarray(jnp.abs(pstar_nn - pstar_true))
    metrics["abs_absolute_median"] = float(np.nanmedian(abs_absolute))
    metrics["abs_absolute_p95"]    = float(np.nanpercentile(abs_absolute, 95.0))
    metrics["abs_absolute_p5"]     = float(np.nanpercentile(abs_absolute, 5.0))

    metrics["any_nan_nn"]   = "true" if bool(jnp.any(jnp.isnan(pstar_nn)))   else "false"
    metrics["any_nan_true"] = "true" if bool(jnp.any(jnp.isnan(pstar_true))) else "false"
    metrics["any_neg_nn"]   = "true" if bool(jnp.any(pstar_nn < 0))          else "false"
    metrics["any_neg_true"] = "true" if bool(jnp.any(pstar_true < 0))        else "false"
    return metrics
