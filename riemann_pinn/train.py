"""Training primitives: samplers, losses, Experiment/Phase runner,
checkpoint I/O, eval. All in 3D delta-space."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from copy import deepcopy

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
from .physics import GAS_STATE_DIM
from .data import DataSet, Sampler, UniformSampler  # noqa: F401


# --- losses ------------------------------------------------------------------


def residual_loss(params, apply_fn, gas_states, targets):
    """Mean squared f(p*) residual. Returns scalar loss."""
    resid_nn = apply_fn({"params": params}, gas_states)
    loss = jnp.mean((resid_nn - targets) ** 2)
    return loss


def supervised_loss(params, apply_fn, gas_states, targets):
    """Mean squared error vs the exact solver p*_true."""
    pstar_nn = apply_fn({"params": params}, gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
    loss = jnp.mean((pstar_nn - pstar_true) ** 2)
    return loss


# --- experiment types --------------------------------------------------------


@dataclass
class Phase:
    """One training phase.

    tx: optax gradient transformation (Adam, L-BFGS, ...).
    loss: (params, apply_fn, x) -> (scalar_loss, metrics_dict).
    sampler: (rng, batch_size, *, drho_range, dp_range, du_range) -> (B, 3).
    fixed_batch=True keeps one sampled batch for the whole phase (typical for L-BFGS).
    is_lbfgs=True selects the L-BFGS call convention in the step function.
    """
    tx: optax.GradientTransformation
    n_epochs: int
    loss: Callable
    batch_size: int
    sampler: DataSet | Sampler
    fixed_batch: bool = False
    is_lbfgs: bool = False
    log_every: int = 200
    name: str = "phase"


@dataclass
class Experiment:
    """One training experiment.

    domain: sampling + evaluation region. Keys drho_range, dp_range, du_range.
    train_domain: optional override used for sampling during training only.
    corner_every: step interval for the optional corner-trace callback.
    output_root: optional override for the parent directory of this run's
        output folder. If unset, defaults to outputs/<file_stem>/.
    """
    name: str
    model: nn.Module
    domain: dict
    phases: list[Phase]
    seed: int = 42
    train_domain: dict | None = None
    corner_every: int = 100
    output_root: str | Path | None = None
    prev_stages: list[tuple[flax_train_state.TrainState, float]] = field(default_factory=list)
    state: flax_train_state.TrainState | None = None

    def evaluate_all_stages(self, gas_states):
        if len(self.prev_stages) == 0:
            return self.state.apply_fn({"params": self.state.params}, gas_states)
        
        all_stages = deepcopy(self.prev_stages)
        all_stages.append((self.state, jnp.nan))
        first_state = all_stages[0][0]
        pstar_nn = first_state.apply_fn({"params": first_state.params}, gas_states)
        for i in range(1, len(all_stages)):
            state = all_stages[i][0]
            eps = all_stages[i-1][1]
            corr_i = state.apply_fn({"params": state.params}, gas_states)
            pstar_nn = pstar_nn * corr_i * eps
        
        return pstar_nn

# --- train state + steps -----------------------------------------------------


def create_train_state(rng, model, tx, batch_size_hint: int = 256):
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(loss_fn: Callable, *, is_lbfgs: bool = False) -> Callable:
    """JITted train step. Branches on is_lbfgs for the optax L-BFGS call convention."""
    if not is_lbfgs:
        @jax.jit
        def step(state, x, targets):
            loss, grads = jax.value_and_grad(
                lambda p: loss_fn(p, state.apply_fn, x, targets),
            )(state.params)
            return state.apply_gradients(grads=grads), loss
        return step

    @jax.jit
    def step(state, x, targets):
        loss, grads = jax.value_and_grad(
            lambda p: loss_fn(p, state.apply_fn, x, targets),
        )(state.params)
        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params,
            value=loss, grad=grads, value_fn=loss,
        )
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(
            step=state.step + 1, params=new_params, opt_state=new_opt_state,
        ), loss
    return step


# --- runner ------------------------------------------------------------------


def run_phase(
    state, phase: Phase, rng, domain: dict,
    *, desc: str | None = None,
    corner_callback: Callable | None = None,
    corner_every: int = 100, step_offset: int = 0,
):
    """Run one phase. Returns (state, loss_trace)."""
    step_fn = make_train_step(phase.loss, is_lbfgs=phase.is_lbfgs)
    loss_trace: list[float] = []
    pbar = tqdm(range(phase.n_epochs), desc=desc or phase.name)
    batch_key = None
    for epoch in pbar:
        if phase.fixed_batch:
            if batch_key is None:
                rng, batch_key = jr.split(rng)
        else:
            rng, batch_key = jr.split(rng)

        if isinstance(phase.sampler, Sampler):
            batch = phase.sampler.draw_batch(batch_key, phase.batch_size)
            targets = None
        elif isinstance(phase.sampler, DataSet):
            batch, targets = phase.sampler.draw_batch(phase.batch_size)
        else:
            raise ValueError(f"Unsupported sampler type: {type(phase.sampler)}")

        state, loss = step_fn(state, batch, targets)
        loss_trace.append(float(loss))
        if epoch % phase.log_every == 0:
            pbar.set_postfix(loss=f"{loss:.2e}")
        if corner_callback is not None:
            gstep = step_offset + epoch + 1
            if gstep % corner_every == 0:
                corner_callback(state, gstep)
    return state, jnp.array(loss_trace)


def run_experiment(exp: Experiment, *, corner_callback: Callable | None = None):
    """Run all phases sequentially. Returns (state, full_loss_trace, per_phase_traces)."""
    rng = jr.PRNGKey(exp.seed)
    state = None
    traces: list[jnp.ndarray] = []
    step_offset = 0
    train_domain = exp.train_domain if exp.train_domain is not None else exp.domain
    n_phases = len(exp.phases)
    for i, phase in enumerate(exp.phases):
        if state is None:
            state = create_train_state(rng, exp.model, phase.tx, batch_size_hint=phase.batch_size)
        else:
            state = flax_train_state.TrainState.create(
                apply_fn=state.apply_fn, params=state.params, tx=phase.tx,
            )
        phase_rng = jr.fold_in(rng, i + 1)
        state, trace = run_phase(
            state, phase, phase_rng, train_domain,
            desc=f"[{i + 1}/{n_phases}] {phase.name}",
            corner_callback=corner_callback, corner_every=exp.corner_every,
            step_offset=step_offset,
        )
        traces.append(trace)
        step_offset += phase.n_epochs
    full_trace = jnp.concatenate(traces) if traces else jnp.array([])
    exp.state = state
    return state, full_trace, traces


def build_template_state(exp: Experiment):
    """Template TrainState used for deserializing a checkpoint of this experiment."""
    rng = jr.PRNGKey(exp.seed)
    last = exp.phases[-1]
    return create_train_state(rng, exp.model, last.tx, batch_size_hint=last.batch_size)


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


def evaluate_holdout(exp: Experiment, n_samples: int = 20_000, seed: int = 999, **domain_kwargs):
    """Residual + pressure-error metrics on a uniform holdout batch."""
    rng = jr.PRNGKey(seed)
    sampler = UniformSampler(**domain_kwargs)
    gas_states = sampler.draw_batch(rng, n_samples)

    pstar_nn = exp.evaluate_all_stages(gas_states)

    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states)

    metrics: dict[str, Any] = {}
    abs_f = jnp.abs(fstar_vals)
    metrics["median_abs_fstar"] = float(jnp.median(abs_f))
    metrics["p95_abs_fstar"] = float(jnp.percentile(abs_f, 95.0))

    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
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
