# riemaNN Simplify/Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `riemann_pinn` to a flat, minimal API (`Experiment`, `Phase`, one `PressureMLP`, callable losses, callable samplers), split `run.py` into `run.py` (train) + `report.py` (metrics), and port every active experiment to the new surface. Drop primary/correction and `StarPressureDS` along the way.

**Architecture:** No string registries, no phase factories, no single-vs-list dichotomy. Every experiment file exports `experiments = [Experiment(name=..., model=..., domain=..., phases=[Phase(...)])]`. `run.py` iterates that list; `report.py` tabulates metrics.json on disk.

**Tech Stack:** Python 3.12, JAX, Flax, optax, NumPy, matplotlib, tqdm.

**Spec:** `docs/superpowers/specs/2026-04-22-simplify-refactor-design.md` (commit `aff1d5c`).

---

## File structure

### New / modified
- `riemann_pinn/model.py` — **rewrite** (~60 lines): `_MLP`, `PressureMLP(normalize="none"|"arith"|"geom")`.
- `riemann_pinn/train.py` — **rewrite** (~230 lines): samplers, 4 losses, `Experiment`, `Phase`, `create_train_state`, `make_train_step`, `run_phase`, `run_experiment`, `build_template_state`, checkpoint I/O, `evaluate_holdout`.
- `run.py` — **rewrite** (~90 lines): train CLI. Inlines the minimal `load_experiments` (no single/list dichotomy).
- `report.py` — **create** (~90 lines): metrics table CLI.
- `CLAUDE.md` — **modify**: update Architecture section.

### Deleted
- `riemann_pinn/experiment.py`
- `riemann_pinn/loader.py`
- `run_experiments.sh`

### Archived (moved to `experiments/archive/`)
- `experiments/adamw_normmlp_geom_correction.py`
- `experiments/lbfgs_normmlp_geom_correction.py`
- `experiments/ds_supervised.py`

### Rewritten in place (14 experiment files)
- Singletons → single-element `experiments` lists: `adam_then_lbfgs.py`, `adam_then_lbfgs_normloss.py`, `adam_then_lbfgs_normmlp.py`, `adam_then_lbfgs_normmlp_100k.py`, `adam_then_lbfgs_normmlp_newton.py`, `adamw.py`, `adamw_normmlp.py`, `lbfgs_ladder.py`, `lbfgs_small.py`, `smoke_test.py`.
- Already list-valued: `adamw_normmlp_geommean.py`, `adamw_normmlp_geom_widen.py`, `adamw_normmlp_geom_widen_L6.py`, `adamw_normmlp_wdgrid.py`.

### Unchanged
- `riemann_pinn/physics.py`, `riemann_pinn/plot.py`, `riemann_pinn/__init__.py`.
- `check_vacuum.py` (its only import, `uniform_log`, survives).
- `run_grid.sh` (minor one-line edit to final command — covered in Task 4).

## Testing strategy

No existing test suite. Validation is a two-pronged smoke approach:

1. **Import-level smoke check** after each module rewrite: a `venv/bin/python -c "..."` one-liner that imports the new surface, instantiates a `PressureMLP`, and runs one forward pass. Fast, catches breakage immediately.
2. **End-to-end smoke run** after porting `smoke_test.py` (Task 5): `venv/bin/python run.py experiments/smoke_test.py --retrain`. This exercises the full pipeline — sampler, loss, two-phase training with Adam→L-BFGS, checkpoint I/O, evaluation, and all plots — in ~30 seconds.
3. **Grid integration check** at the end (Task 9): `venv/bin/python run.py experiments/adamw_normmlp_wdgrid.py --count` (should print `27`), then sampled-index retrain + `report.py` to verify the metrics path.

Every task ends with a commit. Prefix: `refactor:` for all commits in this plan.

---

## Task 1: Archive removed experiments, delete stale scripts

**Files:**
- Move: `experiments/adamw_normmlp_geom_correction.py` → `experiments/archive/`
- Move: `experiments/lbfgs_normmlp_geom_correction.py` → `experiments/archive/`
- Move: `experiments/ds_supervised.py` → `experiments/archive/`
- Delete: `run_experiments.sh`

- [ ] **Step 1: Move the three experiment files into the archive folder**

```bash
git mv experiments/adamw_normmlp_geom_correction.py experiments/archive/adamw_normmlp_geom_correction.py
git mv experiments/lbfgs_normmlp_geom_correction.py experiments/archive/lbfgs_normmlp_geom_correction.py
git mv experiments/ds_supervised.py experiments/archive/ds_supervised.py
```

- [ ] **Step 2: Delete the stale shell script**

```bash
git rm run_experiments.sh
```

Why stale: it references `experiments/*.toml` files and a `--config` flag that don't exist.

- [ ] **Step 3: Verify**

```bash
ls experiments/archive/ | grep -E '(correction|ds_supervised)'
test ! -e run_experiments.sh && echo "deleted"
```

Expected:
```
adamw_normmlp_geom_correction.py
ds_supervised.py
lbfgs_normmlp_geom_correction.py
deleted
```

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: archive primary/correction and DS experiments

- Move adamw_normmlp_geom_correction.py, lbfgs_normmlp_geom_correction.py,
  ds_supervised.py to experiments/archive/. They exercise the
  primary/correction and Deep-Set paths that are being removed.
- Delete stale run_experiments.sh (references nonexistent .toml configs)."
```

---

## Task 2: Rewrite `riemann_pinn/model.py`

**Files:**
- Modify (full rewrite): `riemann_pinn/model.py`

- [ ] **Step 1: Replace `riemann_pinn/model.py` with the new content**

```python
"""Neural network models that predict the star-region pressure p*.

A single `PressureMLP` with a `normalize` flag covers the three variants we
actually use (plain, arithmetic-mean-normalized, geometric-mean-normalized).
Output is physical p* so callers don't need to undo the log transform.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from . import physics


class _MLP(nn.Module):
    """Shared MLP block."""
    width: int
    depth: int
    output_dim: int = 1
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = self.activation(nn.Dense(self.width)(x))
        return nn.Dense(self.output_dim)(x)


class PressureMLP(nn.Module):
    """Maps log-space gas state (B, 5) -> scalar p*.

    normalize:
      "none"   — raw log inputs; network emits log10 p* directly.
      "arith"  — divide by arithmetic means: rho_ref = 0.5*(rhoL+rhoR),
                 p_ref = 0.5*(pL+pR), u_ref = c_s(p_ref, rho_ref).
      "geom"   — geometric means in log space (antisymmetric under L<->R).
    """
    width: int = 64
    depth: int = 2
    normalize: str = "none"
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        model = _MLP(width=self.width, depth=self.depth,
                     activation=self.activation, output_dim=1)

        if self.normalize == "none":
            return 10.0 ** model(x).squeeze(-1)

        if self.normalize == "arith":
            gas_phys = physics.gas_log_to_phys(x)
            rhoL, pL, rhoR, pR, uRL = jnp.split(gas_phys, [1, 2, 3, 4], axis=-1)
            rho_ref = 0.5 * (rhoL + rhoR)
            p_ref = 0.5 * (pL + pR)
            u_ref = physics.sound_speed(p_ref, rho_ref)
            gas_phys_norm = jnp.concatenate(
                [rhoL / rho_ref, pL / p_ref, rhoR / rho_ref, pR / p_ref, uRL / u_ref],
                axis=-1,
            )
            x_norm = physics.gas_phys_to_log(gas_phys_norm)
            log_pstar_over_pref = model(x_norm).squeeze(-1)
            return p_ref.squeeze(-1) * (10.0 ** log_pstar_over_pref)

        if self.normalize == "geom":
            log_rhoL, log_pL, log_rhoR, log_pR, uRL = jnp.split(x, 5, axis=-1)
            log_rho_ref = 0.5 * (log_rhoL + log_rhoR)
            log_p_ref = 0.5 * (log_pL + log_pR)
            p_ref = 10.0 ** log_p_ref
            rho_ref = 10.0 ** log_rho_ref
            u_ref = physics.sound_speed(p_ref, rho_ref)
            x_norm = jnp.concatenate(
                [
                    log_rhoL - log_rho_ref,
                    log_pL - log_p_ref,
                    log_rhoR - log_rho_ref,
                    log_pR - log_p_ref,
                    uRL / u_ref,
                ],
                axis=-1,
            )
            log_pstar_over_pref = model(x_norm).squeeze(-1)
            return p_ref.squeeze(-1) * (10.0 ** log_pstar_over_pref)

        raise ValueError(f"unknown normalize mode: {self.normalize!r}")
```

- [ ] **Step 2: Smoke-check all three normalize modes instantiate and run a forward pass**

```bash
JAX_PLATFORMS=cpu venv/bin/python -c '
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
from riemann_pinn.model import PressureMLP

rng = jr.PRNGKey(0)
x = jnp.ones((4, 5))
for mode in ("none", "arith", "geom"):
    m = PressureMLP(width=8, depth=2, normalize=mode)
    params = m.init(rng, x)["params"]
    y = m.apply({"params": params}, x)
    assert y.shape == (4,), (mode, y.shape)
    assert jnp.all(y > 0), (mode, y)
    print(f"ok {mode}: {y}")
'
```

Expected: three `ok none/arith/geom: [...]` lines with positive floats. Failure here means the rewrite broke the model.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/model.py
git commit -m "refactor: collapse model.py to one PressureMLP + _MLP block

Replaces StarPressureMLP, StarPressureMLPNormalized,
StarPressureMLPNormalizedGeom, PstarLogCorrectionMLP, and StarPressureDS
with a single PressureMLP class parameterized by normalize=
\"none\"|\"arith\"|\"geom\". Drops the multi-output StarPressureMLP
variant (no live experiment used it)."
```

---

## Task 3: Rewrite `riemann_pinn/train.py`, delete `experiment.py` and `loader.py`

**Files:**
- Modify (full rewrite): `riemann_pinn/train.py`
- Delete: `riemann_pinn/experiment.py`
- Delete: `riemann_pinn/loader.py`

- [ ] **Step 1: Replace `riemann_pinn/train.py` with the new content**

```python
"""Training primitives: samplers, losses, Experiment/Phase runner, checkpoint I/O, eval."""

from __future__ import annotations

from dataclasses import dataclass
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
from .physics import GAS_STATE_DIM


# --- samplers ----------------------------------------------------------------


def uniform_log(
    rng,
    batch_size: int,
    *,
    log_rho_range: tuple[float, float] = (0.0, 2.0),
    log_p_range: tuple[float, float] = (-2.0, 2.0),
    u_range: tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Uniform i.i.d. samples in (log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)."""
    keys = jr.split(rng, 5)
    logrhoL = jr.uniform(keys[0], (batch_size,), minval=log_rho_range[0], maxval=log_rho_range[1])
    logpL   = jr.uniform(keys[1], (batch_size,), minval=log_p_range[0],   maxval=log_p_range[1])
    logrhoR = jr.uniform(keys[2], (batch_size,), minval=log_rho_range[0], maxval=log_rho_range[1])
    logpR   = jr.uniform(keys[3], (batch_size,), minval=log_p_range[0],   maxval=log_p_range[1])
    uRL     = jr.uniform(keys[4], (batch_size,), minval=u_range[0],       maxval=u_range[1])
    return jnp.stack([logrhoL, logpL, logrhoR, logpR, uRL], axis=-1)


# R2 quasirandom additive recurrence. Golden ratios for d=1..12.
_R2_GOLDEN = jnp.array([
    1.6180339887498949, 1.3247179572447463, 1.2207440846057596,
    1.1673039782614185, 1.1347241384015194, 1.1127756842787053,
    1.0969815577985598, 1.0850702454914507, 1.0757660660868371,
    1.0682971889208415, 1.0621691678642553, 1.0570505752212287,
])


def r2_quasirandom(
    rng,
    batch_size: int,
    *,
    log_rho_range: tuple[float, float],
    log_p_range: tuple[float, float],
    u_range: tuple[float, float],
) -> jnp.ndarray:
    """R2 quasirandom samples in (log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)."""
    NDIM = 5
    g = _R2_GOLDEN[NDIM - 1]
    powers = jnp.arange(1, NDIM + 1, dtype=jnp.float32)
    a = g ** (-powers)
    x0 = jr.uniform(rng, (NDIM,), minval=0.0, maxval=1.0)
    n = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    out_unit = jnp.mod(x0[None, :] + n * a[None, :], 1.0)
    lo = jnp.array(
        [log_rho_range[0], log_p_range[0], log_rho_range[0], log_p_range[0], u_range[0]],
        dtype=out_unit.dtype,
    )
    hi = jnp.array(
        [log_rho_range[1], log_p_range[1], log_rho_range[1], log_p_range[1], u_range[1]],
        dtype=out_unit.dtype,
    )
    return lo + (hi - lo) * out_unit


# --- losses ------------------------------------------------------------------


def residual_loss(params, apply_fn, gas_states_log):
    """Mean squared f(p*) residual. Returns (loss, metrics)."""
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    loss = jnp.mean(fstar_vals ** 2)
    return loss, {"loss/fstar": loss}


def residual_loss_normalized(params, apply_fn, gas_states_log):
    """Mean of (f(p*) / c_ref)^2, where c_ref is the average sound speed."""
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    cref_vals = jax.vmap(physics.ref_sound_speed)(gas_states_phys)
    loss = jnp.mean((fstar_vals / cref_vals) ** 2)
    return loss, {"loss/fstar": loss}


def residual_loss_newton(params, apply_fn, gas_states_log):
    """Mean of (f/f')^2 — a proxy for squared pressure error without calling the exact solver."""
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    fprime_vals = jax.vmap(physics.dfstar_dp)(pstar, gas_states_phys)
    weight = jax.lax.stop_gradient(fprime_vals)
    loss = jnp.mean((fstar_vals / weight) ** 2)
    return loss, {"loss/newton": loss}


def supervised_loss(params, apply_fn, gas_states_log):
    """Mean squared error vs the exact solver p*_true."""
    pstar_nn = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    loss = jnp.mean((pstar_nn - pstar_true) ** 2)
    return loss, {"loss/supervised": loss}


# --- experiment types --------------------------------------------------------


@dataclass
class Phase:
    """One training phase.

    tx: optax gradient transformation (Adam, L-BFGS, ...).
    loss: (params, apply_fn, x) -> (scalar_loss, metrics_dict).
    sampler: (rng, batch_size, **domain) -> (B, 5) log-space batch.
    fixed_batch=True keeps one sampled batch for the whole phase (typical for L-BFGS).
    is_lbfgs=True selects the L-BFGS call convention in the step function.
    """
    tx: optax.GradientTransformation
    n_epochs: int
    loss: Callable
    batch_size: int = 2048
    sampler: Callable = uniform_log
    fixed_batch: bool = False
    is_lbfgs: bool = False
    log_every: int = 200
    name: str = "phase"


@dataclass
class Experiment:
    """One training experiment.

    domain: sampling + evaluation region. Keys log_rho_range, log_p_range, u_range.
    train_domain: optional override used for sampling during training only.
    corner_every: step interval for the optional corner-trace callback.
    """
    name: str
    model: nn.Module
    domain: dict
    phases: list[Phase]
    seed: int = 42
    train_domain: dict | None = None
    corner_every: int = 100


# --- train state + steps -----------------------------------------------------


def create_train_state(rng, model, tx, batch_size_hint: int = 256):
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(loss_fn: Callable, *, is_lbfgs: bool = False) -> Callable:
    """JITted train step. Branches on is_lbfgs for the optax L-BFGS call convention."""
    if not is_lbfgs:
        @jax.jit
        def step(state, x):
            (loss, metrics), grads = jax.value_and_grad(
                lambda p: loss_fn(p, state.apply_fn, x), has_aux=True,
            )(state.params)
            return state.apply_gradients(grads=grads), loss, metrics
        return step

    @jax.jit
    def step(state, x):
        def value_fn(p):
            l, _ = loss_fn(p, state.apply_fn, x)
            return l
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: loss_fn(p, state.apply_fn, x), has_aux=True,
        )(state.params)
        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params,
            value=loss, grad=grads, value_fn=value_fn,
        )
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(
            step=state.step + 1, params=new_params, opt_state=new_opt_state,
        ), loss, metrics
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
        batch = phase.sampler(batch_key, phase.batch_size, **domain)
        state, loss, _metrics = step_fn(state, batch)
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


def evaluate_holdout(state, n_samples: int = 20_000, seed: int = 999, **domain_kwargs):
    """Residual + pressure-error metrics on a uniform-log holdout batch."""
    rng = jr.PRNGKey(seed)
    gas_states_log = uniform_log(rng, n_samples, **domain_kwargs)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states_phys)

    metrics: dict[str, Any] = {}
    abs_f = jnp.abs(fstar_vals)
    metrics["median_abs_fstar"] = float(jnp.median(abs_f))
    metrics["p95_abs_fstar"] = float(jnp.percentile(abs_f, 95.0))

    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
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
```

- [ ] **Step 2: Delete `riemann_pinn/experiment.py` and `riemann_pinn/loader.py`**

```bash
git rm riemann_pinn/experiment.py riemann_pinn/loader.py
```

- [ ] **Step 3: Smoke-check the new surface**

```bash
JAX_PLATFORMS=cpu venv/bin/python -c '
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr, optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import (
    Experiment, Phase, residual_loss, supervised_loss,
    residual_loss_normalized, residual_loss_newton,
    uniform_log, r2_quasirandom, run_experiment, evaluate_holdout,
    build_template_state, save_checkpoint, load_checkpoint,
)

exp = Experiment(
    name="tiny",
    model=PressureMLP(width=8, depth=2, normalize="geom"),
    domain=dict(log_rho_range=(0.0, 1.0), log_p_range=(0.0, 1.0), u_range=(-0.2, 0.2)),
    seed=0,
    phases=[
        Phase(tx=optax.adam(1e-3), n_epochs=5, loss=residual_loss,
              batch_size=32, log_every=5),
    ],
)
state, trace, _ = run_experiment(exp)
print("final loss:", float(trace[-1]))

# sampler sanity
x1 = uniform_log(jr.PRNGKey(0), 4, **exp.domain)
x2 = r2_quasirandom(jr.PRNGKey(0), 4, **exp.domain)
assert x1.shape == (4, 5) and x2.shape == (4, 5)
print("samplers ok")

m = evaluate_holdout(state, n_samples=200, **exp.domain)
assert "median_abs_fstar" in m
print("evaluate_holdout ok")
'
```

Expected: `final loss: <small number>`, `samplers ok`, `evaluate_holdout ok`.

- [ ] **Step 4: Verify neither old module is importable anywhere**

```bash
JAX_PLATFORMS=cpu venv/bin/python -c '
try:
    import riemann_pinn.experiment  # noqa: F401
except ImportError as e:
    print("ok experiment.py gone:", e)
try:
    import riemann_pinn.loader  # noqa: F401
except ImportError as e:
    print("ok loader.py gone:", e)
'
```

Expected: two `ok ... gone:` lines.

- [ ] **Step 5: Commit**

```bash
git add riemann_pinn/train.py riemann_pinn/experiment.py riemann_pinn/loader.py
git commit -m "refactor: fold Experiment/Phase into train.py, drop factories

- Rewrite train.py: samplers, 4 losses (residual_loss,
  residual_loss_normalized, residual_loss_newton, supervised_loss),
  Experiment/Phase dataclasses, train step (single is_lbfgs branch),
  run_phase / run_experiment runners, ckpt I/O, evaluate_holdout.
- Delete experiment.py (string registries, phase factories,
  PrimarySpec/correction plumbing) and loader.py (single/list dual path).
- Rename R2_quasirandom -> r2_quasirandom and
  residual_loss_supervised -> supervised_loss. Drop residual_loss_allfstar
  (unused). Drop create_train_state_from_apply (primary-only)."
```

---

## Task 4: Rewrite `run.py`, create `report.py`, update `run_grid.sh`

**Files:**
- Modify (full rewrite): `run.py`
- Create: `report.py`
- Modify: `run_grid.sh` (one-line edit)

- [ ] **Step 1: Replace `run.py` with the new content**

```python
"""Run a PINN training experiment defined by a Python file.

Experiment files must export `experiments = [Experiment(...), ...]`
(single-element lists are fine). Outputs land in
outputs/<file_stem>/<exp.name>/.
"""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn.plot import (
    plot_corner_error, plot_corner_pstar, plot_loss,
    plot_pstar_hist2d, plot_slice,
)
from riemann_pinn.train import (
    Experiment, build_template_state, evaluate_holdout, load_checkpoint,
    load_loss_trace, run_experiment, save_checkpoint, save_loss_trace,
)


def load_experiments(path: Path) -> list[Experiment]:
    """Load `experiments = [Experiment(...), ...]` from a Python file."""
    spec = importlib.util.spec_from_file_location("_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "experiments"):
        raise AttributeError(
            f"{path}: must define `experiments = [Experiment(...), ...]`"
        )
    exps = module.experiments
    if not isinstance(exps, (list, tuple)) or not exps:
        raise TypeError(
            f"{path}: `experiments` must be a non-empty list of Experiment instances"
        )
    for i, e in enumerate(exps):
        if not isinstance(e, Experiment):
            raise TypeError(
                f"{path}: experiments[{i}] must be an Experiment instance, "
                f"got {type(e).__name__}"
            )
        if not e.name:
            raise ValueError(
                f"{path}: experiments[{i}] must set `name=...`"
            )
    names = [e.name for e in exps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate names in `experiments`: {names}")
    return list(exps)


def _train_and_eval(
    exp: Experiment, exp_path: Path, out_dir: Path, name: str,
    *, retrain: bool, skip_plots: bool, plot_corner_trace: bool,
) -> None:
    ckpt_path = out_dir / "checkpoint.msgpack"
    loss_path = out_dir / "loss.npy"
    metrics_path = out_dir / "metrics.json"
    plots_dir = out_dir / "plots"

    training_time_s = None
    if ckpt_path.is_file() and not retrain:
        print(f"Loading checkpoint from {ckpt_path}")
        state = load_checkpoint(ckpt_path, build_template_state(exp))
        loss_trace = load_loss_trace(loss_path)
    else:
        frames_dir = plots_dir / "corner_frames"

        def corner_cb(s, step):
            plot_corner_error(
                s, frames_dir / f"corner_error_{step:07d}.png",
                name=f"{name} @ step {step}", **exp.domain,
            )

        cb = corner_cb if plot_corner_trace else None
        t0 = time.monotonic()
        state, loss_trace, _ = run_experiment(exp, corner_callback=cb)
        training_time_s = round(time.monotonic() - t0, 1)
        save_checkpoint(ckpt_path, state)
        save_loss_trace(loss_path, loss_trace)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_path, out_dir / exp_path.name)

    metrics = evaluate_holdout(state, **exp.domain)
    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if skip_plots:
        return
    if loss_trace is not None:
        plot_loss(loss_trace, plots_dir / "loss.png", title=f"Training loss — {name}")
    plot_slice(
        state, plots_dir / "slice.png",
        log_rho_range=exp.domain["log_rho_range"],
        log_p_range=exp.domain["log_p_range"],
        name=name,
    )
    plot_pstar_hist2d(state, plots_dir / "pstar_hist2d.png", name=name, **exp.domain)
    plot_corner_error(state, plots_dir / "corner_error.png", name=name, **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", name=name, **exp.domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="train only experiments[N] (default: all)")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--plot-corner-trace", action="store_true")
    ap.add_argument("--count", action="store_true",
                    help="print len(experiments) and exit")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    if args.count:
        print(len(exps))
        return

    if args.index is not None:
        if not 0 <= args.index < len(exps):
            raise IndexError(
                f"{exp_path}: --index {args.index} out of range [0, {len(exps)})"
            )
        selected = [exps[args.index]]
    else:
        selected = exps

    for exp in selected:
        out_dir = out_root / stem / exp.name
        name = f"{stem}/{exp.name}"
        _train_and_eval(
            exp, exp_path, out_dir, name,
            retrain=args.retrain, skip_plots=args.skip_plots,
            plot_corner_trace=args.plot_corner_trace,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `report.py`**

```python
"""Print metrics tables for trained experiments.

Reads outputs/<file_stem>/<exp.name>/metrics.json for each experiment in
the given file and prints a compact table. Missing runs show 'nr'.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from run import load_experiments


_METRIC_LABELS: dict[str, str] = {
    "median_abs_fstar":          "med|f(p*)|",
    "p95_abs_fstar":             "p95|f(p*)|",
    "median_abs_delta_log10_p":  "med|dlog10p*|",
    "p95_abs_delta_log10_p":     "p95|dlog10p*|",
    "abs_absolute_median":       "med|dp*|",
    "abs_absolute_p5":           "p5|dp*|",
    "abs_absolute_p95":          "p95|dp*|",
    "any_nan_nn":                "nan_nn",
    "any_nan_true":              "nan_tr",
    "any_neg_nn":                "neg_nn",
    "any_neg_true":              "neg_tr",
    "training_time_s":           "t (s)",
}


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "nr"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    loaded = [
        _load_metrics(out_root / stem / e.name / "metrics.json") for e in exps
    ]

    if len(exps) == 1:
        m = loaded[0]
        if m is None:
            print(f"{stem}/{exps[0].name}: nr (no metrics.json in "
                  f"{out_root / stem / exps[0].name})")
            return
        width = max(len(_METRIC_LABELS.get(k, k)) for k in m) if m else 0
        for k in sorted(m):
            label = _METRIC_LABELS.get(k, k)
            print(f"{label:<{width}}  {_fmt_cell(m[k])}")
        return

    metric_keys: set[str] = set()
    for m in loaded:
        if m is not None:
            metric_keys.update(m.keys())
    # drop all-false boolean-like columns for readability
    for k in list(metric_keys):
        vals = {m[k] for m in loaded if m is not None and k in m}
        if vals and vals.issubset({"false"}):
            metric_keys.discard(k)
    metric_cols = sorted(metric_keys)
    if "training_time_s" in metric_cols:
        metric_cols.remove("training_time_s")
        metric_cols.append("training_time_s")

    header = ["name"] + [_METRIC_LABELS.get(k, k) for k in metric_cols]
    rows: list[list[str]] = []
    for e, m in zip(exps, loaded):
        if m is None:
            row = [e.name] + ["nr"] * len(metric_cols)
        else:
            row = [e.name] + [_fmt_cell(m.get(k)) for k in metric_cols]
        rows.append(row)

    _print_table(rows, header)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `run_grid.sh`**

Find the final line (originally `venv/bin/python run.py "$EXPERIMENT" --print-metrics`) and replace with `venv/bin/python report.py "$EXPERIMENT"`. Use this edit:

```
Old: venv/bin/python run.py "$EXPERIMENT" --print-metrics
New: venv/bin/python report.py "$EXPERIMENT"
```

- [ ] **Step 4: Smoke-check CLI parsing**

```bash
venv/bin/python run.py --help | head -5
venv/bin/python report.py --help | head -5
```

Expected: both print usage lines without error.

- [ ] **Step 5: Commit**

```bash
git add run.py report.py run_grid.sh
git commit -m "refactor: split CLI into run.py (train) and report.py (metrics)

- run.py: always expects \`experiments = [Experiment(...), ...]\`; trains
  all in order or one selected by --index N. Keeps --retrain,
  --skip-plots, --plot-corner-trace, --count. Drops --print-metrics.
- report.py (new): reads outputs/<stem>/<name>/metrics.json and prints
  a table. Single-experiment files get a key-value listing; lists get a
  table.
- run_grid.sh: final metrics call points at report.py. --count still
  served by run.py."
```

---

## Task 5: Port `smoke_test.py` and run it end-to-end

This is the integration canary. Run it before touching the other experiment files.

**Files:**
- Modify: `experiments/smoke_test.py`

- [ ] **Step 1: Replace `experiments/smoke_test.py` with the ported version**

```python
"""Tiny experiment for smoke-testing the pipeline (few epochs)."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=32, depth=2, normalize="none"),
        domain=dict(
            log_rho_range=(-2.0, 2.0),
            log_p_range=(-2.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=0,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(1e-3, 20, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=20,
                loss=residual_loss,
                batch_size=64,
                log_every=10,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=10,
                loss=residual_loss,
                batch_size=64,
                fixed_batch=True,
                is_lbfgs=True,
                log_every=5,
                name="lbfgs",
            ),
        ],
    ),
]
```

- [ ] **Step 2: Wipe any stale checkpoint under the new output path**

```bash
rm -rf outputs/smoke_test
```

- [ ] **Step 3: Run end-to-end with retrain**

```bash
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/smoke_test.py --retrain
```

Expected: training progress bars for two phases, a metrics JSON printed to stdout (containing `median_abs_fstar`, `p95_abs_fstar`, etc.), and plot files written under `outputs/smoke_test/main/plots/`.

- [ ] **Step 4: Confirm artifacts**

```bash
ls outputs/smoke_test/main/
ls outputs/smoke_test/main/plots/
```

Expected under `outputs/smoke_test/main/`: `checkpoint.msgpack`, `loss.npy`, `metrics.json`, `smoke_test.py`, `plots/`.
Expected under `plots/`: `loss.png`, `slice.png`, `pstar_hist2d.png`, `corner_error.png`, `corner_pstar.png`.

- [ ] **Step 5: Confirm `report.py` works on a single-experiment file**

```bash
venv/bin/python report.py experiments/smoke_test.py
```

Expected: a vertical listing of metric labels and values (not a wide table, because it's a single experiment).

- [ ] **Step 6: Commit**

```bash
git add experiments/smoke_test.py
git commit -m "refactor: port smoke_test.py to new Experiment/Phase API

Pipeline canary for the refactor. Verified end-to-end: training runs,
checkpoint saves/loads, metrics.json writes, all five plot kinds
generated, report.py prints the single-experiment summary."
```

---

## Task 6: Port the five `adam_then_lbfgs*` experiments

**Files:**
- Modify: `experiments/adam_then_lbfgs.py`
- Modify: `experiments/adam_then_lbfgs_normmlp.py`
- Modify: `experiments/adam_then_lbfgs_normmlp_100k.py`
- Modify: `experiments/adam_then_lbfgs_normmlp_newton.py`
- Modify: `experiments/adam_then_lbfgs_normloss.py`

**Translation rules used throughout this task (applied identically to Tasks 7 and 8):**

| Old | New |
|-----|-----|
| `experiment = Experiment(...)` | `experiments = [Experiment(name="main", ...)]` |
| `StarPressureMLP(width, depth)` | `PressureMLP(width, depth, normalize="none")` |
| `StarPressureMLPNormalized(width, depth)` | `PressureMLP(width, depth, normalize="arith")` |
| `StarPressureMLPNormalizedGeom(width, depth)` | `PressureMLP(width, depth, normalize="geom")` |
| `loss="fstar"` | `loss=residual_loss` |
| `loss="fstar_normalized"` | `loss=residual_loss_normalized` |
| `loss="newton"` | `loss=residual_loss_newton` |
| `loss="pstar"` | `loss=supervised_loss` |
| `sampler="uniform"` | `sampler=uniform_log` (default, usually omit) |
| `sampler="r2"` | `sampler=r2_quasirandom` |
| `adam_cosine(n_epochs=N, lr=L, alpha=A, batch_size=B, loss=X, log_every=E)` | `Phase(tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(L, N, alpha=A), weight_decay=1e-4)), n_epochs=N, loss=X, batch_size=B, log_every=E, name="adam_cosine")`. When the old call omitted `alpha=`, substitute `alpha=0.0` in the expansion (`optax.cosine_decay_schedule`'s own default). |
| `adam(n_epochs=N, lr=L, batch_size=B, loss=X, log_every=E)` | `Phase(tx=optax.adam(L), n_epochs=N, loss=X, batch_size=B, log_every=E, name="adam")` |
| `lbfgs(n_epochs=N, batch_size=B, loss=X, log_every=E, memory_size=M, learning_rate=LR, name=N, sampler=S)` | `Phase(tx=optax.lbfgs(learning_rate=LR, memory_size=M), n_epochs=N, loss=X, batch_size=B, log_every=E, fixed_batch=True, is_lbfgs=True, sampler=...as-needed, name=...as-needed)`. Keep `memory_size=10`, `learning_rate=1.0` unless the original overrode. |

All `adam_cosine` in the old code used `weight_decay=1e-4` and `clip_norm=1.0` (the factory defaults) — preserve those in the expansion. Name each Phase so log lines stay informative (`"adam_cosine"`, `"lbfgs"`, etc., matching the old phase default or any custom `name=` in the old call).

- [ ] **Step 1: Rewrite `experiments/adam_then_lbfgs.py`**

```python
"""MLP: Adam warmup + two L-BFGS phases at different batch sizes."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(-2.0, 2.0),
            log_p_range=(-2.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(1e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=256,
                log_every=200,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=1_000,
                loss=residual_loss,
                batch_size=256,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b256",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=500,
                loss=residual_loss,
                batch_size=1_024,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b1024",
            ),
        ],
    ),
]
```

- [ ] **Step 2: Rewrite `experiments/adam_then_lbfgs_normmlp.py`**

```python
"""Normalized MLP: Adam warmup (Adam-cosine only; L-BFGS phase was commented out)."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-0.5, 0.5),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=1e-4),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 3: Rewrite `experiments/adam_then_lbfgs_normmlp_100k.py`**

```python
"""Normalized MLP: 100k Adam-cosine, no L-BFGS."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 100_000, alpha=1e-6),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=100_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=20,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 4: Rewrite `experiments/adam_then_lbfgs_normmlp_newton.py`**

```python
"""Normalized MLP + Newton-step loss ((f/f')^2) on narrowed u domain."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss_newton


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-0.5, 0.5),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=1e-4),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss_newton,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 5: Rewrite `experiments/adam_then_lbfgs_normloss.py`**

Note: old file kept `corner_every=200`. Preserve it.

```python
"""MLP with the normalized (f/c_ref)^2 loss."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss_normalized


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        corner_every=200,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=1e-6),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss_normalized,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 6: Verify all five files load without error**

```bash
for f in \
  experiments/adam_then_lbfgs.py \
  experiments/adam_then_lbfgs_normmlp.py \
  experiments/adam_then_lbfgs_normmlp_100k.py \
  experiments/adam_then_lbfgs_normmlp_newton.py \
  experiments/adam_then_lbfgs_normloss.py; do
  JAX_PLATFORMS=cpu venv/bin/python run.py "$f" --count
done
```

Expected: each prints `1` (one experiment per file).

- [ ] **Step 7: Commit**

```bash
git add experiments/adam_then_lbfgs.py experiments/adam_then_lbfgs_normmlp.py \
        experiments/adam_then_lbfgs_normmlp_100k.py \
        experiments/adam_then_lbfgs_normmlp_newton.py \
        experiments/adam_then_lbfgs_normloss.py
git commit -m "refactor: port adam_then_lbfgs* experiments to new API

Singleton experiment files become single-element experiments lists named
'main'. adam_cosine(...) expands to an inline optax.chain(clip_by_global_norm,
adamw(cosine_decay_schedule, wd=1e-4)), lbfgs(...) expands to
optax.lbfgs() with fixed_batch=True, is_lbfgs=True. Loss strings become
imported callables."
```

---

## Task 7: Port the remaining four singletons (`adamw*`, `lbfgs_*`)

**Files:**
- Modify: `experiments/adamw.py`
- Modify: `experiments/adamw_normmlp.py`
- Modify: `experiments/lbfgs_ladder.py`
- Modify: `experiments/lbfgs_small.py`

- [ ] **Step 1: Rewrite `experiments/adamw.py`**

```python
"""MLP: 10k Adam-cosine baseline on the f* residual."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 2: Rewrite `experiments/adamw_normmlp.py`**

```python
"""Normalized MLP: 10k Adam-cosine baseline on the f* residual."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 3: Rewrite `experiments/lbfgs_ladder.py`**

Original has three phases: 10k Adam-cosine, then two L-BFGS phases at batch 256 and 1024. (Confirmed from the current file content.)

```python
"""MLP: Adam warmup, then two L-BFGS phases at different batch sizes."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(-2.0, 2.0),
            log_p_range=(-2.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(1e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=256,
                log_every=200,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=1_000,
                loss=residual_loss,
                batch_size=256,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b256",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=500,
                loss=residual_loss,
                batch_size=1_024,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b1024",
            ),
        ],
    ),
]
```

- [ ] **Step 4: Rewrite `experiments/lbfgs_small.py`**

Original: 5k Adam-cosine (lr=8e-3, alpha=0.0, bs=2048) + 100k L-BFGS (bs=8192, sampler=r2) on `PressureMLP(width=8, depth=6, normalize="geom")`. Keep the `lbfgs_w8_d6` name.

```python
"""Narrow/deep normalized MLP with L-BFGS on R2 quasirandom samples."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import (
    Experiment, Phase, residual_loss, r2_quasirandom,
)


experiments = [
    Experiment(
        name="lbfgs_w8_d6",
        model=PressureMLP(width=8, depth=6, normalize="geom"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(8e-3, 5_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=5_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=100_000,
                loss=residual_loss,
                batch_size=8_192,
                fixed_batch=True,
                is_lbfgs=True,
                sampler=r2_quasirandom,
                log_every=200,
                name="lbfgs",
            ),
        ],
    ),
]
```

- [ ] **Step 5: Verify all four files load**

```bash
for f in experiments/adamw.py experiments/adamw_normmlp.py \
         experiments/lbfgs_ladder.py experiments/lbfgs_small.py; do
  JAX_PLATFORMS=cpu venv/bin/python run.py "$f" --count
done
```

Expected: each prints `1`.

- [ ] **Step 6: Commit**

```bash
git add experiments/adamw.py experiments/adamw_normmlp.py \
        experiments/lbfgs_ladder.py experiments/lbfgs_small.py
git commit -m "refactor: port adamw*/lbfgs_* singleton experiments to new API"
```

---

## Task 8: Port the four list-valued experiments

**Files:**
- Modify: `experiments/adamw_normmlp_wdgrid.py`
- Modify: `experiments/adamw_normmlp_geommean.py`
- Modify: `experiments/adamw_normmlp_geom_widen.py`
- Modify: `experiments/adamw_normmlp_geom_widen_L6.py`

- [ ] **Step 1: Rewrite `experiments/adamw_normmlp_wdgrid.py`**

```python
"""Width/depth/lr grid over the normalized MLP baseline (adamw_normmlp).

Same optimizer, loss, domain, seed, and phase budget as adamw_normmlp.py;
only width/depth/lr vary. Select one with --index N; outputs land in
outputs/adamw_normmlp_wdgrid/<name>/.
"""

from itertools import product

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)

_WIDTHS = [64, 128, 256]
_DEPTHS = [2, 3, 4]
_LRS    = [2e-3, 4e-3, 8e-3]


def _make(width: int, depth: int, lr: float) -> Experiment:
    return Experiment(
        name=f"w{width}_d{depth}_lr{lr}",
        model=PressureMLP(width=width, depth=depth, normalize="arith"),
        domain=_DOMAIN,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(lr, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    )


experiments = [_make(w, d, lr) for w, d, lr in product(_WIDTHS, _DEPTHS, _LRS)]
```

- [ ] **Step 2: Rewrite `experiments/adamw_normmlp_geommean.py`**

```python
"""Arithmetic vs geometric mean for the normalized-MLP reference scales.

Base config is w64_d2_lr0.008 of adamw_normmlp_wdgrid.py; only the
normalize= mode changes ("arith" vs "geom"). Geometric is the arithmetic
mean in log space and is antisymmetric under L<->R, which may make the
learning problem easier.
"""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)


def _phases():
    return [
        Phase(
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    optax.cosine_decay_schedule(8e-3, 10_000, alpha=0.0),
                    weight_decay=1e-4,
                ),
            ),
            n_epochs=10_000,
            loss=residual_loss,
            batch_size=2048,
            log_every=200,
            name="adam_cosine",
        ),
    ]


experiments = [
    Experiment(
        name="arithmetic",
        model=PressureMLP(width=64, depth=2, normalize="arith"),
        domain=_DOMAIN, seed=42, phases=_phases(),
    ),
    Experiment(
        name="geometric",
        model=PressureMLP(width=64, depth=2, normalize="geom"),
        domain=_DOMAIN, seed=42, phases=_phases(),
    ),
]
```

- [ ] **Step 3: Rewrite `experiments/adamw_normmlp_geom_widen.py`**

```python
"""Effect of widening the training domain beyond the evaluation domain.

Narrow: train_domain == domain (training = evaluation region).
Wide:   train_domain extended by +/- 0.2 in log rho/log p and +/- 0.1 in
        uRL, so some training samples fall outside the test region.

The wide run scales its batch size by the 5-D volume ratio so per-batch
sample density is held constant. Evaluation/plot domain is the same in
both runs.
"""

from math import prod

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


_BASE_BATCH_SIZE = 2048

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)
_WIDE_TRAIN_DOMAIN = dict(
    log_rho_range=(-0.2, 2.2),
    log_p_range=(-0.2, 2.2),
    u_range=(-0.6, 0.6),
)


def _volume(dom: dict) -> float:
    widths = {k: v[1] - v[0] for k, v in dom.items()}
    return prod([
        widths["log_rho_range"], widths["log_p_range"],
        widths["log_rho_range"], widths["log_p_range"],
        widths["u_range"],
    ])


def _scaled_batch(train_domain: dict) -> int:
    ratio = _volume(train_domain) / _volume(_DOMAIN)
    return int(round(_BASE_BATCH_SIZE * ratio))


def _phases(batch_size: int):
    return [
        Phase(
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    optax.cosine_decay_schedule(8e-3, 40_000, alpha=0.0),
                    weight_decay=1e-4,
                ),
            ),
            n_epochs=40_000,
            loss=residual_loss,
            batch_size=batch_size,
            log_every=200,
            name="adam_cosine",
        ),
    ]


def _make(name: str, train_domain: dict | None) -> Experiment:
    bs = _BASE_BATCH_SIZE if train_domain is None else _scaled_batch(train_domain)
    return Experiment(
        name=name,
        model=PressureMLP(width=64, depth=2, normalize="geom"),
        domain=_DOMAIN,
        train_domain=train_domain,
        seed=42,
        phases=_phases(bs),
    )


experiments = [
    _make("narrow", None),
    _make("wide", _WIDE_TRAIN_DOMAIN),
]
```

- [ ] **Step 4: Rewrite `experiments/adamw_normmlp_geom_widen_L6.py`**

This one uses a custom (params, apply_fn, x)-signature loss that closes over a per-experiment `l6` weight. Keep that pattern; it's exactly what the new API supports.

```python
"""Mixed p*-MSE + L6 penalty atop adamw_normmlp_geom_widen.

Same model, domain (wide train_domain), and budget as the widen experiment;
sweeps the l6 weight in a mean_sq + l6 * mean_abs6 supervised loss.
"""

from math import prod

import jax
import jax.numpy as jnp
import optax

from riemann_pinn import physics
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase


_BASE_BATCH_SIZE = 2048

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)
_WIDE_TRAIN_DOMAIN = dict(
    log_rho_range=(-0.2, 2.2),
    log_p_range=(-0.2, 2.2),
    u_range=(-0.6, 0.6),
)


def _volume(dom: dict) -> float:
    widths = {k: v[1] - v[0] for k, v in dom.items()}
    return prod([
        widths["log_rho_range"], widths["log_p_range"],
        widths["log_rho_range"], widths["log_p_range"],
        widths["u_range"],
    ])


def _scaled_batch(train_domain: dict) -> int:
    ratio = _volume(train_domain) / _volume(_DOMAIN)
    return int(round(_BASE_BATCH_SIZE * ratio))


def _make_loss(l6: float):
    def loss_fn(params, apply_fn, gas_states_log):
        pstar_nn = apply_fn({"params": params}, gas_states_log)
        gas_states_phys = physics.gas_log_to_phys(gas_states_log)
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
        mse = jnp.mean((pstar_nn - pstar_true) ** 2)
        l6_term = jnp.mean(jnp.abs((pstar_nn - pstar_true) ** 6))
        loss = (1.0 - l6) * mse + l6 * l6_term
        return loss, {"loss": loss}
    return loss_fn


def _make(name: str, train_domain: dict | None, l6: float) -> Experiment:
    assert 0.0 <= l6 <= 1.0
    bs = _BASE_BATCH_SIZE if train_domain is None else _scaled_batch(train_domain)
    return Experiment(
        name=name,
        model=PressureMLP(width=64, depth=2, normalize="geom"),
        domain=_DOMAIN,
        train_domain=train_domain,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(8e-3, 20_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=20_000,
                loss=_make_loss(l6),
                batch_size=bs,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    )


experiments = [
    _make("l6_0.0", _WIDE_TRAIN_DOMAIN, 0.0),
    _make("l6_0.1", _WIDE_TRAIN_DOMAIN, 0.1),
    _make("l6_0.4", _WIDE_TRAIN_DOMAIN, 0.4),
]
```

- [ ] **Step 5: Verify all four files load with the correct counts**

```bash
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/adamw_normmlp_wdgrid.py --count      # expect 27
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/adamw_normmlp_geommean.py --count    # expect 2
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/adamw_normmlp_geom_widen.py --count  # expect 2
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/adamw_normmlp_geom_widen_L6.py --count  # expect 3
```

- [ ] **Step 6: Commit**

```bash
git add experiments/adamw_normmlp_wdgrid.py experiments/adamw_normmlp_geommean.py \
        experiments/adamw_normmlp_geom_widen.py experiments/adamw_normmlp_geom_widen_L6.py
git commit -m "refactor: port list-valued experiments (wdgrid, geommean, geom_widen, L6)

The L6 variant keeps its custom closure-based loss — the new API allows
any Callable with (params, apply_fn, x) signature, so no workaround is
needed."
```

---

## Task 9: End-to-end validation

This task does NOT modify code. If anything fails, fix it (find the minimal change, add a new commit; do not amend earlier commits).

- [ ] **Step 1: Re-run `smoke_test` one more time to confirm the full port didn't break it**

```bash
rm -rf outputs/smoke_test
JAX_PLATFORMS=cpu venv/bin/python run.py experiments/smoke_test.py --retrain
```

Expected: completes in <60s, writes `outputs/smoke_test/main/metrics.json`, and prints the metrics JSON to stdout.

- [ ] **Step 2: Confirm `report.py` on `smoke_test`**

```bash
venv/bin/python report.py experiments/smoke_test.py
```

Expected: vertical key-value listing of the metric labels (`med|f(p*)|`, `p95|f(p*)|`, etc.) with numeric values.

- [ ] **Step 3: Confirm `--count` works for each active experiment file**

```bash
for f in experiments/*.py; do
  [[ -d "$f" ]] && continue
  count=$(JAX_PLATFORMS=cpu venv/bin/python run.py "$f" --count 2>/dev/null || echo FAIL)
  printf '%-55s %s\n' "$f" "$count"
done
```

Expected (every non-archive file prints an integer; the `archive/` dir is skipped):
```
experiments/adam_then_lbfgs.py                          1
experiments/adam_then_lbfgs_normloss.py                 1
experiments/adam_then_lbfgs_normmlp.py                  1
experiments/adam_then_lbfgs_normmlp_100k.py             1
experiments/adam_then_lbfgs_normmlp_newton.py           1
experiments/adamw.py                                    1
experiments/adamw_normmlp.py                            1
experiments/adamw_normmlp_geom_widen.py                 2
experiments/adamw_normmlp_geom_widen_L6.py              3
experiments/adamw_normmlp_geommean.py                   2
experiments/adamw_normmlp_wdgrid.py                     27
experiments/lbfgs_ladder.py                             1
experiments/lbfgs_small.py                              1
experiments/smoke_test.py                               1
```

If any prints `FAIL`, open that file and cross-check against the translation rules in Task 6. Fix with a focused commit titled `refactor: fix <file> port`.

- [ ] **Step 4: Confirm `run_grid.sh` still drives the pipeline**

Run with a small parallelism setting. Use `--count` path first as a dry-run (no training), then exercise an index on a fast file:

```bash
./run_grid.sh experiments/adamw_normmlp_geommean.py 2
```

Expected: 2-parallel training of the 2 experiments, ending with the `report.py` metrics table showing both experiments and their metrics. This trains ~20k Adam steps per experiment, so it takes a few minutes — OK to skip on CPU if that's unreasonable for your machine; note so in the step notes and move on.

- [ ] **Step 5: No commit (validation-only task). If Step 4 was skipped due to time, say so inline.**

---

## Task 10: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace the `## Architecture` and `## Running` sections**

Open `CLAUDE.md` and replace the `## Running`, `## Architecture`, and `## Key Conventions` sections (keep `## Project Overview` and `## Notebooks` unchanged) with:

```markdown
## Running

```bash
venv/bin/python run.py experiments/smoke_test.py                 # train + evaluate + plot
venv/bin/python run.py experiments/smoke_test.py --retrain       # ignore checkpoint, train from scratch
venv/bin/python run.py experiments/smoke_test.py --skip-plots    # skip plot generation
venv/bin/python run.py experiments/foo.py --index 2              # list-valued file: train only experiments[2]
venv/bin/python run.py experiments/foo.py --count                # print number of experiments
venv/bin/python report.py experiments/foo.py                     # print metrics table
./run_grid.sh experiments/foo.py 4                               # fan out a list over 4 workers + report
```

Outputs go to `outputs/<file_stem>/<exp.name>/`: `checkpoint.msgpack`,
`loss.npy`, `metrics.json`, `plots/loss.png`, `plots/slice.png`,
`plots/pstar_hist2d.png`, `plots/corner_error.png`, `plots/corner_pstar.png`.

Note: if `jax-metal` crashes, set `JAX_PLATFORMS=cpu` before running.

## Architecture

```
riemann_pinn/
    physics.py   — Riemann solver: GAMMA, fjump, fstar, dfstar_dp, find_pstar, gas_log_to_phys, sound_speed
    model.py     — _MLP block + PressureMLP(normalize="none"|"arith"|"geom")
    train.py     — samplers (uniform_log, r2_quasirandom), losses (residual_loss, residual_loss_normalized,
                   residual_loss_newton, supervised_loss), Experiment/Phase dataclasses, run_phase/run_experiment,
                   checkpoint I/O, evaluate_holdout
    plot.py      — plot_loss, plot_slice, plot_corner_error, plot_corner_pstar, plot_pstar_hist2d
run.py           — CLI: load experiments list, train, save metrics, plot
report.py        — CLI: read metrics.json files, print table
```

Experiment files export `experiments = [Experiment(...), ...]` — always a
list (one-element lists are fine). Each `Experiment` has `name`, `model`,
`domain`, `phases`, `seed` (and optional `train_domain`, `corner_every`).
Each `Phase` has `tx` (an `optax.GradientTransformation`), `n_epochs`,
`loss` (a callable), plus `batch_size`, `sampler`, `fixed_batch`,
`is_lbfgs`, `log_every`, `name`.

## Key Conventions

- `jax.config.update("jax_enable_x64", True)` must be set before JAX ops (done in `run.py`).
- Gas states are length-5: `(rhoL, pL, rhoR, pR, uRL)` in physical space, or `(log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)` in log space. `GAS_STATE_DIM = 5` lives in `physics.py`.
- Network input is log-space; training domain is `log_rho in [-2, 2]`, `log_p in [-2, 2]`, `uRL in [-1, 1]` (overridable per experiment).
- Output artifacts are gitignored.
- Losses have the signature `(params, apply_fn, gas_states_log) -> (scalar_loss, metrics_dict)`. Custom losses live in the experiment file that needs them and are passed via `Phase(loss=...)`.
- Checkpoints are saved with the last phase's optimizer shape; changing the tail phase's `tx` type invalidates existing checkpoints.
```

- [ ] **Step 2: Verify markdown looks right**

```bash
head -80 CLAUDE.md
```

Expected: `## Project Overview`, the new `## Running`, the new `## Architecture`, the new `## Key Conventions`, and the untouched `## Notebooks` section.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for the new flat Experiment/Phase API"
```

---

## Appendix: spec coverage checklist

Quick self-check that every spec requirement has a task:

- Target module layout (new model.py, new train.py, new run.py + report.py; delete experiment.py, loader.py, run_experiments.sh) — **Tasks 1, 2, 3, 4**
- `Experiment` / `Phase` dataclasses with listed fields — **Task 3**
- Example experiment file under new API — **Task 5** (smoke_test.py is the canonical example)
- `PressureMLP(normalize=...)` collapses 3 classes — **Task 2**
- Delete `PstarLogCorrectionMLP`, `StarPressureDS` — **Task 2** (rewrite)
- Losses kept/renamed: `residual_loss`, `residual_loss_normalized`, `residual_loss_newton`, `supervised_loss` — **Task 3**
- Drop `residual_loss_allfstar` — **Task 3** (absent from new file)
- Samplers kept/renamed: `uniform_log`, `r2_quasirandom` — **Task 3**
- `make_train_step(loss, is_lbfgs=...)` — **Task 3**
- `run_phase`, `run_experiment` — **Task 3**
- `create_train_state` kept, `create_train_state_from_apply` dropped — **Task 3**
- Checkpoint / loss trace I/O, `evaluate_holdout` kept — **Task 3**
- `run.py` CLI flags: `--index`, `--retrain`, `--skip-plots`, `--plot-corner-trace`, `--count`; no `--print-metrics` — **Task 4**
- `report.py` reads metrics.json files, tabulates, handles single-experiment case — **Task 4**
- `run_grid.sh` minimal edit — **Task 4**
- Always `outputs/<file_stem>/<exp.name>/` — **Task 4** (`_train_and_eval` uses that path)
- Archive the 3 experiments, delete `run_experiments.sh` — **Task 1**
- Rewrite the 14 active experiment files — **Tasks 5, 6, 7, 8**
- Validation: smoke_test e2e, list file --count, run_grid.sh — **Task 9**
- Update CLAUDE.md Architecture section — **Task 10**
- Breaking: old checkpoints unloadable (retrain mitigation) — called out in Task 10 key convention + Task 5 Step 2 (`rm -rf outputs/smoke_test`).
