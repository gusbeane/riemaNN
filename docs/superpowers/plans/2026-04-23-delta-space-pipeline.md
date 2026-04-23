# Delta-space pipeline — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the training pipeline from its 5D `(log₁₀ρ_L, log₁₀p_L, log₁₀ρ_R, log₁₀p_R, u_RL)` representation to a 3D `(drho, dp, du)` representation everywhere, cutting redundant physics/model/loss paths on the way.

**Architecture:** Single representation — delta-space with `p_ref = 1, ρ_ref = 1` implicit. `fstar`/`ftilde`/`get_ducrit` in `physics.py` are already 3D-native; the work is pushing that representation out to samplers, model input, losses, plots, evaluation, and the one live experiment, and deleting everything that only existed to bridge to 5D.

**Tech Stack:** Python 3.12, JAX 0.9.2, Flax 0.12.6, optax 0.2.8, numpy 2.4.4, matplotlib 3.10.8. No test framework — use `venv/bin/python -c '...'` assertion snippets as the verification pattern (matches the existing ad-hoc-script style in `check_vacuum.py` and `test_prime.py`).

**Reference spec:** `docs/superpowers/specs/2026-04-23-delta-space-pipeline-design.md`.

**Working directory:** `/home/abeane/Projects/riemaNN/.worktrees/dimens-reduc`. All commands in this plan assume that is the cwd. All file paths given below are relative to it.

**Pre-existing state:** The worktree has uncommitted changes to `riemann_pinn/physics.py` — the 3D `fstar` / `ftilde` / `get_ducrit` kernel is already in place but `two_rarefaction_p0`, `gas_log_to_phys`, etc. still coexist. Task 1 finishes that migration in a single commit; do not commit the worktree's current intermediate state separately.

---

## File Structure After the Refactor

- `riemann_pinn/physics.py` — 3D Riemann primitives: `GAMMA`/`ALPHA`/`BETA`/`MU`/`GAS_STATE_DIM=3`, `get_ducrit`, `ftilde`, `fstar`, `dfstar_dp`, `_newton`, `_bisect`, `find_pstar`, `two_rarefaction_p0`. Nothing 5D.
- `riemann_pinn/model.py` — one `PressureMLP` class (3D → log-space `p*/p_ref`) + private `_MLP` block.
- `riemann_pinn/train.py` — `uniform` and `r2_quasirandom` samplers (both return `(B, 3)`), three losses (`residual_loss`, `residual_loss_newton`, `supervised_loss`), `Experiment`/`Phase` dataclasses, train step builder, phase/experiment runner, checkpoint I/O, `evaluate_holdout`.
- `riemann_pinn/plot.py` — `plot_loss` (unchanged), `plot_slice` (3-panel `(drho, dp)` at fixed `du`), `plot_pstar_hist2d`, `_corner_panels` (3 pairs), `plot_corner_error`, `plot_corner_pstar`.
- `run.py` — CLI driver; only the `plot_slice` call site changes.
- `experiments/smoke_test.py` — fresh single-phase AdamW experiment using the new domain.
- `experiments/archive/*` — frozen historical files; not expected to import after this refactor.
- `CLAUDE.md` — rewritten Architecture + Key Conventions sections.
- Deleted: `experiments_hipres/`, `check_vacuum.py`, `test_prime.py`.

---

## Task 1: Finalize `riemann_pinn/physics.py`

**Files:**
- Modify: `riemann_pinn/physics.py` (complete rewrite; absorbs the uncommitted 3D kernel + deletes 5D helpers + adds 3D `two_rarefaction_p0`)

**Pre-state check:**

- [ ] **Step 1: Confirm the uncommitted 3D kernel is present in the worktree**

Run: `git diff riemann_pinn/physics.py | grep -E 'def (ftilde|get_ducrit|fstar)' | head`
Expected: shows `+def get_ducrit(drho, dp):`, `+def ftilde(p, drho, dp, LR):`, `-def fstar(pstar, gas_state):` / `+def fstar(p, gas_state):`. If those lines are absent, stop and surface the discrepancy — the worktree is in a different state than the plan assumes.

**Rewrite:**

- [ ] **Step 2: Replace the full contents of `riemann_pinn/physics.py` with the final 3D form**

Write the following exact contents to `riemann_pinn/physics.py`:

```python
"""Riemann problem physics for an ideal gas with gamma = 5/3, 3D form.

A "gas state" is a length-3 array `(drho, dp, du)`:

  drho = (rhoR - rhoL) / (rhoR + rhoL)  in [-1, 1]
  dp   = (pR   - pL)   / (pR   + pL)    in [-1, 1]
  du   = uRL / ducrit(drho, dp)         in [-inf, 1]

The non-dimensionalization p_ref = 1, rho_ref = 1 is implicit:
pL = 1 - dp, pR = 1 + dp, rhoL = 1 - drho, rhoR = 1 + drho.
Sound speeds, ducrit, and p* are all dimensionless (ratios to p_ref).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

GAMMA: float = 5.0 / 3.0
ALPHA: float = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA: float = (GAMMA - 1.0) / (GAMMA + 1.0)
MU: float = (GAMMA - 1.0) / 2.0

GAS_STATE_DIM: int = 3


@jax.jit
def get_ducrit(drho, dp):
    ansL = jnp.sqrt((1 + dp) / (1 + drho))
    ansR = jnp.sqrt((1 - dp) / (1 - drho))
    return (2.0 / (GAMMA - 1.0)) * (ansL + ansR)


@jax.jit
def ftilde(p, drho, dp, LR):
    """Contribution of one side (K = L or R) to the Riemann star-pressure equation.

    LR = -1 for L, +1 for R.
    """
    AK = jnp.sqrt(2.0 / (GAMMA * (GAMMA + 1.0) * (1 + LR * drho)))
    BK = BETA * (1 + LR * dp)

    shock = (p - (1 + LR * dp)) * AK / jnp.sqrt(p + BK)
    rarefaction = (1.0 / MU) * jnp.sqrt(
        (1 + LR * dp) / (1 + LR * drho)
    ) * ((p / (1 + LR * dp)) ** ALPHA - 1.0)
    return jnp.where(p > (1 + LR * dp), shock, rarefaction)


@jax.jit
def fstar(p, gas_state):
    """Residual of the Riemann star-pressure equation; zero at the true p*."""
    drho, dp, du = gas_state
    ducrit = get_ducrit(drho, dp)
    return (ftilde(p, drho, dp, -1) + ftilde(p, drho, dp, 1)) / ducrit + du


dfstar_dp = jax.grad(fstar, argnums=0)


@jax.jit
def two_rarefaction_p0(gas_state):
    """3D two-rarefaction p* guess (Toro eq. 4.46), dimensionless.

    Used only as the Newton / bisection starting point inside find_pstar.
    """
    drho, dp, du = gas_state
    cL = jnp.sqrt(GAMMA * (1 - dp) / (1 - drho))
    cR = jnp.sqrt(GAMMA * (1 + dp) / (1 + drho))
    ducrit = get_ducrit(drho, dp)
    num = jnp.maximum(cL + cR - MU * du * ducrit, 1e-30)
    den = cL / (1 - dp) ** ALPHA + cR / (1 + dp) ** ALPHA
    return (num / den) ** (1.0 / ALPHA)


@jax.jit
def _newton(gas_state, p0):
    """Newton iteration for p*. Fast but can diverge."""

    def cond(state):
        pstar, pstar_prev, fstar_, i = state
        return (
            (jnp.abs(fstar_) >= 1e-12)
            & (jnp.abs(pstar - pstar_prev) >= 1e-10)
            & (i < 100)
        )

    def body(state):
        pstar, _pstar_prev, _, i = state
        pstar_prev = pstar
        fstar_ = fstar(pstar, gas_state)
        dfstar_ = dfstar_dp(pstar, gas_state)
        pstar = pstar - fstar_ / dfstar_
        return pstar, pstar_prev, fstar(pstar, gas_state), i + 1

    init = (p0, jnp.inf, fstar(p0, gas_state), 0)
    pstar, _pstar_prev, fstar_final, _ = jax.lax.while_loop(cond, body, init)
    return pstar, fstar_final


@jax.jit
def _bisect(gas_state):
    """Bisection solver for p*. Slow but guaranteed to converge.

    Vacuum states (no physical root) return a very small p* with a
    non-zero residual.
    """
    p_guess = jnp.maximum(two_rarefaction_p0(gas_state), 1e-30)

    p_lo = p_guess * 0.5
    p_hi = p_guess * 2.0
    f_lo = fstar(p_lo, gas_state)
    f_hi = fstar(p_hi, gas_state)

    def widen_cond(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        return (_f_lo * _f_hi > 0) & (i < 60)

    def widen_body(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        _p_lo = jnp.where(_f_lo > 0, jnp.maximum(_p_lo * 0.01, 1e-30), _p_lo)
        _p_hi = jnp.where(_f_hi < 0, _p_hi * 100.0, _p_hi)
        return (
            _p_lo, _p_hi,
            fstar(_p_lo, gas_state), fstar(_p_hi, gas_state), i + 1,
        )

    p_lo, p_hi, f_lo, f_hi, _ = jax.lax.while_loop(
        widen_cond, widen_body, (p_lo, p_hi, f_lo, f_hi, 0)
    )

    def bisect_cond(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        return ((_p_hi - _p_lo) > 1e-12 * jnp.maximum(p_mid, 1e-30)) & (i < 200)

    def bisect_body(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        f_mid = fstar(p_mid, gas_state)
        _p_lo = jnp.where(f_mid < 0, p_mid, _p_lo)
        _p_hi = jnp.where(f_mid >= 0, p_mid, _p_hi)
        return _p_lo, _p_hi, i + 1

    p_lo, p_hi, _ = jax.lax.while_loop(
        bisect_cond, bisect_body, (p_lo, p_hi, 0)
    )

    p_result = 0.5 * (p_lo + p_hi)
    f_result = fstar(p_result, gas_state)
    return p_result, f_result


@jax.jit
def find_pstar(gas_state):
    """Find p* via Newton with bisection fallback; returns (pstar, residual)."""
    p0 = jnp.maximum(two_rarefaction_p0(gas_state), 1e-30)
    p_newton, f_newton = _newton(gas_state, p0)

    newton_ok = (
        jnp.isfinite(p_newton)
        & jnp.isfinite(f_newton)
        & (p_newton > 0)
        & (jnp.abs(f_newton) < 1e-12)
    )

    p_bisect, f_bisect = jax.lax.cond(
        newton_ok,
        lambda _: (p_newton, f_newton),
        lambda _: _bisect(gas_state),
        None,
    )
    return p_bisect, f_bisect
```

**Verify:**

- [ ] **Step 3: Import + basic invariant check**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from riemann_pinn import physics

# Constants
assert physics.GAS_STATE_DIM == 3
assert abs(physics.GAMMA - 5/3) < 1e-12

# Trivial case: identical states (drho=dp=du=0) should give p* = 1 (=p_ref).
gs = jnp.array([0.0, 0.0, 0.0])
p, f = physics.find_pstar(gs)
assert abs(float(p) - 1.0) < 1e-9, f'p*={p}, expected 1.0'
assert abs(float(f)) < 1e-9, f'residual={f}'

# Non-trivial case: fstar(p*, gs) must be ~0 at the returned p*.
gs2 = jnp.array([0.3, -0.2, 0.1])
p2, f2 = physics.find_pstar(gs2)
assert float(p2) > 0 and abs(float(f2)) < 1e-9, (p2, f2)

# L<->R symmetry: swapping sides sends (drho, dp, du) -> (-drho, -dp, -du)
# and leaves p* (a physical pressure) invariant.
gs3 = jnp.array([0.5, -0.3, -0.4])
gs3_swap = -gs3
p3, _ = physics.find_pstar(gs3)
p3s, _ = physics.find_pstar(gs3_swap)
assert abs(float(p3) - float(p3s)) < 1e-8, (p3, p3s)

# No symbol should be importable that was deleted.
for name in ['sound_speed', 'ref_sound_speed', 'find_ustar',
             'gas_log_to_phys', 'gas_phys_to_log',
             'two_rarefaction_p0_batch', 'fjump']:
    assert not hasattr(physics, name), f'{name} still present'

print('physics.py OK')
"
```
Expected stdout: `physics.py OK`. No assertion errors. No import errors.

- [ ] **Step 4: Commit**

```bash
git add riemann_pinn/physics.py
git commit -m "refactor(physics): collapse to 3D delta-state representation

Drop 5D helpers (sound_speed, ref_sound_speed, find_ustar,
gas_log_to_phys, gas_phys_to_log, two_rarefaction_p0_batch, fjump),
rewrite two_rarefaction_p0 natively in (drho, dp, du) space, and set
GAS_STATE_DIM = 3. p_ref = 1, rho_ref = 1 is now implicit."
```

---

## Task 2: Rewrite `riemann_pinn/model.py`

**Files:**
- Modify: `riemann_pinn/model.py` (full rewrite — single `PressureMLP` class)

- [ ] **Step 1: Replace the full contents of `riemann_pinn/model.py`**

Write the following exact contents to `riemann_pinn/model.py`:

```python
"""Neural network model that predicts dimensionless p* = p*/p_ref.

Input is a 3D delta-state (drho, dp, du). Output is log-scale to
guarantee positivity and to match the ~1-2 order-of-magnitude dynamic
range of p*/p_ref over the sampling domain.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp  # noqa: F401 (kept for downstream experimenters)


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
    """Maps (B, 3) delta state -> (B,) p*/p_ref, positive by log output."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):  # x: (B, 3) = (drho, dp, du)
        model = _MLP(
            width=self.width, depth=self.depth,
            activation=self.activation, output_dim=1,
        )
        return 10.0 ** model(x).squeeze(-1)
```

- [ ] **Step 2: Verify init + apply on a dummy 3D batch**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.random as jr
import jax.numpy as jnp
from riemann_pinn.model import PressureMLP

model = PressureMLP(width=32, depth=2)
rng = jr.PRNGKey(0)
dummy = jnp.ones((4, 3))
params = model.init(rng, dummy)['params']
out = model.apply({'params': params}, dummy)
assert out.shape == (4,), f'shape={out.shape}'
assert bool(jnp.all(out > 0)), f'non-positive output: {out}'
print('model.py OK')
"
```
Expected stdout: `model.py OK`.

- [ ] **Step 3: Confirm deleted symbols are gone**

Run: `grep -nE 'CompactDimensionPressureMLP|normalize' riemann_pinn/model.py`
Expected: no output (empty grep → exit 1 is fine).

- [ ] **Step 4: Commit**

```bash
git add riemann_pinn/model.py
git commit -m "refactor(model): collapse to single 3D PressureMLP

Delete 5D PressureMLP (all normalize modes) and
CompactDimensionPressureMLP. The new PressureMLP takes (B, 3) delta
input and emits p*/p_ref via 10**model(x), giving positivity by
construction and matching the dynamic range of the target."
```

---

## Task 3: Rewrite `riemann_pinn/train.py`

**Files:**
- Modify: `riemann_pinn/train.py` (full rewrite of samplers, losses, and `evaluate_holdout`; runner machinery is unchanged)

- [ ] **Step 1: Replace the full contents of `riemann_pinn/train.py`**

Write the following exact contents to `riemann_pinn/train.py`:

```python
"""Training primitives: samplers, losses, Experiment/Phase runner,
checkpoint I/O, eval. All in 3D delta-space."""

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


def uniform(
    rng,
    batch_size: int,
    *,
    drho_range: tuple[float, float],
    dp_range: tuple[float, float],
    du_range: tuple[float, float],
) -> jnp.ndarray:
    """Uniform i.i.d. samples in (drho, dp, du). Returns (B, 3)."""
    keys = jr.split(rng, 3)
    drho = jr.uniform(keys[0], (batch_size,), minval=drho_range[0], maxval=drho_range[1])
    dp   = jr.uniform(keys[1], (batch_size,), minval=dp_range[0],   maxval=dp_range[1])
    du   = jr.uniform(keys[2], (batch_size,), minval=du_range[0],   maxval=du_range[1])
    return jnp.stack([drho, dp, du], axis=-1)


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
    drho_range: tuple[float, float],
    dp_range: tuple[float, float],
    du_range: tuple[float, float],
) -> jnp.ndarray:
    """R2 quasirandom samples in (drho, dp, du). Returns (B, 3)."""
    NDIM = 3
    g = _R2_GOLDEN[NDIM - 1]
    powers = jnp.arange(1, NDIM + 1, dtype=jnp.float32)
    a = g ** (-powers)
    x0 = jr.uniform(rng, (NDIM,), minval=0.0, maxval=1.0)
    n = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    out_unit = jnp.mod(x0[None, :] + n * a[None, :], 1.0)
    lo = jnp.array([drho_range[0], dp_range[0], du_range[0]], dtype=out_unit.dtype)
    hi = jnp.array([drho_range[1], dp_range[1], du_range[1]], dtype=out_unit.dtype)
    return lo + (hi - lo) * out_unit


# --- losses ------------------------------------------------------------------


def residual_loss(params, apply_fn, gas_states):
    """Mean squared f(p*) residual. Returns (loss, metrics)."""
    pstar = apply_fn({"params": params}, gas_states)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states)
    loss = jnp.mean(fstar_vals ** 2)
    return loss, {"loss/fstar": loss}


def residual_loss_newton(params, apply_fn, gas_states):
    """Mean of (f/f')^2 -- a proxy for squared pressure error."""
    pstar = apply_fn({"params": params}, gas_states)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states)
    fprime_vals = jax.vmap(physics.dfstar_dp)(pstar, gas_states)
    weight = jax.lax.stop_gradient(fprime_vals)
    loss = jnp.mean((fstar_vals / weight) ** 2)
    return loss, {"loss/newton": loss}


def supervised_loss(params, apply_fn, gas_states):
    """Mean squared error vs the exact solver p*_true."""
    pstar_nn = apply_fn({"params": params}, gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
    loss = jnp.mean((pstar_nn - pstar_true) ** 2)
    return loss, {"loss/supervised": loss}


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
    batch_size: int = 2048
    sampler: Callable = uniform
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
    """Residual + pressure-error metrics on a uniform holdout batch."""
    rng = jr.PRNGKey(seed)
    gas_states = uniform(rng, n_samples, **domain_kwargs)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states)
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
```

- [ ] **Step 2: Verify samplers + losses run end-to-end**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
from riemann_pinn import train, model, physics

dom = dict(drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9))

# uniform sampler
rng = jr.PRNGKey(0)
x = train.uniform(rng, 128, **dom)
assert x.shape == (128, 3), x.shape
assert bool(((x[:, 0] >= -0.9) & (x[:, 0] <= 0.9)).all())
assert bool(((x[:, 2] >= -3.0) & (x[:, 2] <= 0.9)).all())

# r2 sampler
x2 = train.r2_quasirandom(rng, 128, **dom)
assert x2.shape == (128, 3)

# losses + model
m = model.PressureMLP(width=16, depth=2)
params = m.init(jr.PRNGKey(1), jnp.ones((1, 3)))['params']
for lfn in (train.residual_loss, train.residual_loss_newton, train.supervised_loss):
    loss, metrics = lfn(params, m.apply, x)
    assert jnp.isfinite(loss), (lfn.__name__, loss)

# GAS_STATE_DIM passes through
assert train.GAS_STATE_DIM == 3

# deleted symbols
for name in ['uniform_log', 'residual_loss_normalized']:
    assert not hasattr(train, name), f'{name} still present'

print('train.py OK')
"
```
Expected stdout: `train.py OK`.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/train.py
git commit -m "refactor(train): 3D samplers, losses, and evaluation

- uniform / r2_quasirandom take drho_range / dp_range / du_range and
  return (B, 3).
- residual_loss / residual_loss_newton / supervised_loss take 3D gas
  states directly; no more gas_log_to_phys bridge.
- residual_loss_normalized deleted (ref_sound_speed normalization is
  redundant once fstar is dimensionless).
- evaluate_holdout samples with uniform() and calls find_pstar / fstar
  on 3D input.
- Phase.sampler default is now uniform."
```

---

## Task 4: Rewrite `riemann_pinn/plot.py`

**Files:**
- Modify: `riemann_pinn/plot.py` (full rewrite)

- [ ] **Step 1: Replace the full contents of `riemann_pinn/plot.py`**

Write the following exact contents to `riemann_pinn/plot.py`:

```python
"""Loss curve, 3D-slice heatmap, hist2d, and corner plots (3 pairs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from . import physics  # noqa: E402
from .train import uniform  # noqa: E402


_VAR_LABELS = [r"$\Delta\rho$", r"$\Delta p$", r"$\Delta u$"]


def plot_loss(loss_trace, out_path: Path, *, title: str = "Training loss") -> None:
    arr = np.asarray(loss_trace)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(np.log10(arr))
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
    ax.set_ylim(-8, 4)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slice(
    state, out_path: Path, *, n: int = 250,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9),
    du_slice: float = 0.0,
    err_range=(-0.1, 0.1), nbins: int = 100, name: str | None = None,
) -> None:
    """Three-panel slice: log10(p_NN/p_true) and signed log|f| over (drho, dp)
    at du = du_slice, plus a histogram of log10(p_NN/p_true) with an L<->R
    swap sanity-check overlay.
    """
    dr = jnp.linspace(*drho_range, n)
    dpv = jnp.linspace(*dp_range, n)
    dr_grid, dp_grid = jnp.meshgrid(dr, dpv, indexing="ij")
    gas_states = jnp.stack([
        dr_grid.ravel(), dp_grid.ravel(),
        jnp.full(n * n, du_slice),
    ], axis=-1)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states)

    log_ratio = np.asarray(jnp.log10(pstar_nn / pstar_true).reshape(n, n))
    signed_f = np.asarray(
        (jnp.sign(fstar_vals) * jnp.log10(jnp.maximum(jnp.abs(fstar_vals), 1e-30)))
        .reshape(n, n)
    )

    dr_np = np.asarray(dr)
    dp_np = np.asarray(dpv)
    titles = {
        "log_ratio": r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        "signed_f": r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, z, key in zip(axes[:2], [log_ratio, signed_f], ["log_ratio", "signed_f"]):
        v = float(np.nanmax(np.abs(z)))
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        c = ax.pcolormesh(dr_np, dp_np, z.T, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_xlabel(_VAR_LABELS[0])
        ax.set_ylabel(_VAR_LABELS[1])
        ax.set_title(titles[key] + f" @ $\\Delta u={du_slice:g}$")
        fig.colorbar(c, ax=ax)

    ax = axes[2]
    bins = np.linspace(err_range[0], err_range[1], nbins)
    ax.hist(log_ratio.ravel(), bins=bins, histtype="step", ec="k", density=True,
            label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$")
    ax.set(xlim=err_range, yscale="log")

    # L<->R symmetry sanity check: the Riemann problem is invariant under
    # (drho, dp, du) -> (-drho, -dp, -du). Plot the NN's deviation from
    # that invariance as a second histogram.
    gas_states_swap = -gas_states
    pstar_nn_swap = state.apply_fn({"params": state.params}, gas_states_swap)
    log_ratio_sym = np.asarray(jnp.log10(pstar_nn) - jnp.log10(pstar_nn_swap))
    ax.hist(log_ratio_sym, bins=bins, histtype="step", ec="b", density=True,
            label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{NN,swap}})$")
    ax.legend(fontsize=8)

    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Corner plots -------------------------------------------------------------


def _corner_panels(n, *, drho_range, dp_range, du_range):
    """Build gas-state grids for all 3 lower-triangle variable pairs over
    (drho, dp, du). Returns (panels, axes_np) where panels is a list of
    (grid_row, grid_col, gas_states) and axes_np[k] is the 1-D numpy array
    for variable k.
    """
    ranges = [drho_range, dp_range, du_range]
    mids = [0.5 * (r[0] + r[1]) for r in ranges]
    axes = [jnp.linspace(r[0], r[1], n) for r in ranges]

    panels = []
    for row in range(2):
        for col in range(row + 1):
            x_var, y_var = col, row + 1
            xg, yg = jnp.meshgrid(axes[x_var], axes[y_var], indexing="ij")
            state_cols = []
            for k in range(3):
                if k == x_var:
                    state_cols.append(xg.ravel())
                elif k == y_var:
                    state_cols.append(yg.ravel())
                else:
                    state_cols.append(jnp.full(n * n, mids[k]))
            panels.append((row, col, jnp.stack(state_cols, axis=-1)))
    return panels, [np.asarray(a) for a in axes]


def plot_corner_error(
    state, out_path: Path, *, n: int = 50,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_NN / p*_true) over all 3 input-variable pairs."""
    panels, axes_np = _corner_panels(
        n, drho_range=drho_range, dp_range=dp_range, du_range=du_range,
    )

    all_gas = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_nn_all = state.apply_fn({"params": state.params}, all_gas)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(all_gas)
    log_ratio_all = np.asarray(jnp.log10(pstar_nn_all / pstar_true_all))

    panel_data = []
    offset = 0
    for row, col, gas in panels:
        npts = gas.shape[0]
        z = log_ratio_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmax = max(float(np.nanmax(np.abs(d[2]))) for d in panel_data)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 1].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="RdBu_r", norm=norm,
        )
        if row < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(_VAR_LABELS[col])
        if col > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(_VAR_LABELS[row + 1])

    fig.colorbar(
        mappable, ax=axarr.ravel().tolist(),
        label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        shrink=0.6, pad=0.02,
    )
    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corner_pstar(
    out_path: Path, *, n: int = 50,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_true) over all 3 input-variable pairs."""
    panels, axes_np = _corner_panels(
        n, drho_range=drho_range, dp_range=dp_range, du_range=du_range,
    )

    all_gas = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(all_gas)
    log_pstar_all = np.asarray(jnp.log10(jnp.maximum(pstar_true_all, 1e-30)))

    panel_data = []
    offset = 0
    for row, col, gas in panels:
        npts = gas.shape[0]
        z = log_pstar_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmin = min(float(np.nanmin(d[2])) for d in panel_data)
    vmax = max(float(np.nanmax(d[2])) for d in panel_data)
    if not np.isfinite(vmin):
        vmin = -2.0
    if not np.isfinite(vmax):
        vmax = 2.0

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 1].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        if row < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(_VAR_LABELS[col])
        if col > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(_VAR_LABELS[row + 1])

    fig.colorbar(
        mappable, ax=axarr.ravel().tolist(),
        label=r"$\log_{10}\,p^*_{\mathrm{true}}$",
        shrink=0.6, pad=0.02,
    )
    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pstar_hist2d(
    state, out_path: Path, *, n_samples: int = 50_000, seed: int = 999,
    nbins: int = 200, name: str | None = None,
    **domain_kwargs,
) -> None:
    """2D histogram of log10(p*_true) vs log10(p*_NN)."""
    import jax.random as jr

    rng = jr.PRNGKey(seed)
    gas_states = uniform(rng, n_samples, **domain_kwargs)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)

    log_true = np.asarray(jnp.log10(jnp.maximum(pstar_true, 1e-30)))
    log_nn = np.asarray(jnp.log10(jnp.maximum(pstar_nn, 1e-30)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    lims = [min(log_true.min(), log_nn.min()), max(log_true.max(), log_nn.max())]
    bins = np.linspace(lims[0], lims[1], nbins)
    h = ax.hist2d(log_true, log_nn, bins=bins, norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="count")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"$\log_{10}\,p^*_{\mathrm{true}}$")
    ax.set_ylabel(r"$\log_{10}\,p^*_{\mathrm{NN}}$")
    title = r"$p^*$ prediction vs truth"
    if name:
        title = f"{title} -- {name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Verify the module imports and all plot entry points exist**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
from riemann_pinn import plot
for name in ['plot_loss', 'plot_slice', 'plot_pstar_hist2d',
             'plot_corner_error', 'plot_corner_pstar', '_corner_panels']:
    assert hasattr(plot, name), name
# Deleted imports (old helpers) must not leak through.
import riemann_pinn.plot as p
src = open(p.__file__).read()
assert 'gas_log_to_phys' not in src
assert 'uniform_log' not in src
print('plot.py OK')
"
```
Expected stdout: `plot.py OK`.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/plot.py
git commit -m "refactor(plot): 3D delta-space plots

plot_slice now uses (drho, dp) at fixed du=du_slice with a histogram
panel overlay comparing (drho, dp, du) vs its L<->R swap
(-drho, -dp, -du) as a sanity check. Corner plots reduce to a 2x2
lower-triangle over the 3 variable pairs. plot_pstar_hist2d samples
in delta-space via train.uniform."
```

---

## Task 5: Update `run.py` call site

**Files:**
- Modify: `run.py` (one call)

- [ ] **Step 1: Update the `plot_slice` call**

Edit `run.py` to change the `plot_slice` block:

```python
    plot_slice(
        state, plots_dir / "slice.png",
        log_rho_range=exp.domain["log_rho_range"],
        log_p_range=exp.domain["log_p_range"],
        name=name,
    )
```

to:

```python
    plot_slice(
        state, plots_dir / "slice.png",
        drho_range=exp.domain["drho_range"],
        dp_range=exp.domain["dp_range"],
        name=name,
    )
```

Nothing else in `run.py` needs to change.

- [ ] **Step 2: Verify import + argparse help still works**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
import run  # noqa: F401
print('run.py OK')
"
venv/bin/python run.py --help > /dev/null
```
Expected: `run.py OK` on stdout, `--help` exits 0 and prints usage to stderr/stdout without error.

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "refactor(run): plot_slice takes drho_range/dp_range"
```

---

## Task 6: Move old experiments to archive, delete stale scripts

**Files:**
- Move: every file directly under `experiments/` (not in a subdirectory) → `experiments/archive/`
- Delete: `experiments_hipres/` (directory)
- Delete: `check_vacuum.py`, `test_prime.py`

- [ ] **Step 1: List the files to move (sanity check)**

Run: `ls experiments/ | grep -E '\.py$'`
Expected output (order may vary):
```
adam_then_lbfgs_normloss.py
adam_then_lbfgs_normmlp_100k.py
adam_then_lbfgs_normmlp_newton.py
adam_then_lbfgs_normmlp.py
adam_then_lbfgs.py
adamw_dreducmlp.py
adamw_normmlp_geommean.py
adamw_normmlp_geom_widen_L6.py
adamw_normmlp_geom_widen.py
adamw_normmlp.py
adamw_normmlp_wdgrid.py
adamw.py
lbfgs_ladder.py
lbfgs_small.py
smoke_test.py
```

- [ ] **Step 2: git-mv every top-level experiment file into archive**

Run:
```bash
for f in adam_then_lbfgs_normloss.py adam_then_lbfgs_normmlp_100k.py \
         adam_then_lbfgs_normmlp_newton.py adam_then_lbfgs_normmlp.py \
         adam_then_lbfgs.py adamw_dreducmlp.py adamw_normmlp_geommean.py \
         adamw_normmlp_geom_widen_L6.py adamw_normmlp_geom_widen.py \
         adamw_normmlp.py adamw_normmlp_wdgrid.py adamw.py \
         lbfgs_ladder.py lbfgs_small.py smoke_test.py; do
  git mv "experiments/$f" "experiments/archive/$f"
done
```
If a file has already been moved (e.g. partial rerun), the `git mv` for that file will fail; rerun after removing those entries from the loop.

- [ ] **Step 3: Delete `experiments_hipres/` and the stale top-level scripts**

Run:
```bash
git rm -r experiments_hipres
git rm check_vacuum.py test_prime.py
```

- [ ] **Step 4: Verify `experiments/` contains only `archive/`**

Run: `ls experiments/`
Expected: `archive` (a directory, nothing else — `smoke_test.py` is not yet recreated; that's Task 7).

- [ ] **Step 5: Verify no live code references deleted symbols**

Run:
```bash
grep -rnE 'gas_log_to_phys|gas_phys_to_log|sound_speed|ref_sound_speed|find_ustar|CompactDimensionPressureMLP|residual_loss_normalized|uniform_log|log_rho_range|log_p_range|\bu_range\b' \
  riemann_pinn/ run.py report.py 2>/dev/null
```
Expected: no output. (Matches in `experiments/archive/` and `note/` are allowed and not searched.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: archive 5D experiments, delete stale 5D scripts

Move every top-level experiments/*.py into experiments/archive/ as
frozen historical record (not expected to import after this
refactor). Delete experiments_hipres/ (single 5D file) and the
one-off check_vacuum.py / test_prime.py exploratory scripts that
used the removed 5D APIs."
```

---

## Task 7: Write new `experiments/smoke_test.py`

**Files:**
- Create: `experiments/smoke_test.py`

- [ ] **Step 1: Write the new experiment file**

Write the following exact contents to `experiments/smoke_test.py`:

```python
"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, supervised_loss, uniform


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.9),
)


experiments = [
    Experiment(
        name="smoke_test",
        model=PressureMLP(width=64, depth=2),
        domain=_DOMAIN,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 1_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=1_000,
                loss=supervised_loss,
                batch_size=2048,
                sampler=uniform,
                log_every=100,
                name="adam_cosine",
            ),
        ],
    ),
]
```

- [ ] **Step 2: Verify the experiment loads without running**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
from pathlib import Path
from run import load_experiments
exps = load_experiments(Path('experiments/smoke_test.py'))
assert len(exps) == 1
exp = exps[0]
assert exp.name == 'smoke_test'
assert set(exp.domain.keys()) == {'drho_range', 'dp_range', 'du_range'}
assert len(exp.phases) == 1
print('smoke_test.py OK')
"
```
Expected stdout: `smoke_test.py OK`.

- [ ] **Step 3: Commit**

```bash
git add experiments/smoke_test.py
git commit -m "feat(experiments): add 3D smoke_test.py

Single-phase AdamW cosine over 1000 epochs with supervised_loss
against find_pstar ground truth, sampling on the default delta-space
domain. Serves as the minimal working example for the new pipeline."
```

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (rewrite Architecture and Key Conventions sections)

- [ ] **Step 1: Rewrite `CLAUDE.md`**

Write the following exact contents to `CLAUDE.md`:

```markdown
# CLAUDE.md

## Project Overview

riemaNN trains a physics-informed neural network (PINN) to predict the star-region pressure `p*/p_ref` from a 3D delta gas state by minimizing the squared residual of the exact Riemann jump function `f(p*)`. Python 3.12, JAX + Flax + optax. Ideal gas, `gamma = 5/3`.

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
    physics.py   — 3D Riemann primitives: GAMMA, ALPHA, BETA, MU, GAS_STATE_DIM (=3),
                   get_ducrit, ftilde, fstar, dfstar_dp, find_pstar, two_rarefaction_p0
    model.py     — _MLP block + PressureMLP (3D input -> log-space p*/p_ref)
    train.py     — samplers (uniform, r2_quasirandom), losses
                   (residual_loss, residual_loss_newton, supervised_loss),
                   Experiment/Phase dataclasses, run_phase/run_experiment,
                   checkpoint I/O, evaluate_holdout
    plot.py      — plot_loss, plot_slice, plot_corner_error, plot_corner_pstar,
                   plot_pstar_hist2d
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
- Gas states are length-3: `(drho, dp, du)` where
  `drho = (rhoR - rhoL)/(rhoR + rhoL) ∈ [-1, 1]`,
  `dp   = (pR   - pL)  /(pR   + pL)   ∈ [-1, 1]`,
  `du   = uRL / ducrit(drho, dp)      ∈ [-∞, 1]`.
  The non-dimensionalization `p_ref = 1, rho_ref = 1` is implicit:
  `pL = 1-dp, pR = 1+dp, rhoL = 1-drho, rhoR = 1+drho`. `GAS_STATE_DIM = 3` lives in `physics.py`.
- Network input is the 3D delta state. Default domain is
  `drho_range = dp_range = (-0.9, 0.9)`, `du_range = (-3.0, 0.9)`
  (overridable per experiment).
- Model output is `10**model(x)`, i.e. `p*/p_ref` in log space — positive by construction.
- Output artifacts are gitignored.
- Losses have the signature `(params, apply_fn, gas_states) -> (scalar_loss, metrics_dict)`. Custom losses live in the experiment file that needs them and are passed via `Phase(loss=...)`.
- Checkpoints are saved with the last phase's optimizer shape; changing the tail phase's `tx` type invalidates existing checkpoints.
- `experiments/archive/` holds frozen 5D-era experiments for historical reference; they are not expected to import or run against the current code.

## Notebooks

`note/0-damp_osc.ipynb` and `note/1-ideal_riemann.ipynb` are standalone explorations — they do not share code with `riemann_pinn/`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): update Architecture and Key Conventions for 3D pipeline"
```

---

## Task 9: End-to-end verification (smoke run)

**Files:** none created or modified — verification only.

- [ ] **Step 1: Confirm the live code paths have no 5D leftovers**

Run:
```bash
grep -rnE 'gas_log_to_phys|gas_phys_to_log|ref_sound_speed|find_ustar|CompactDimensionPressureMLP|residual_loss_normalized|uniform_log|log_rho_range|log_p_range|\bu_range\b|fjump' \
  riemann_pinn/ run.py report.py experiments/smoke_test.py 2>/dev/null
```
Expected: no output.

- [ ] **Step 2: Confirm every public module imports cleanly**

Run:
```bash
venv/bin/python -c "
import jax
jax.config.update('jax_enable_x64', True)
from riemann_pinn import physics, model, train, plot
import run, report  # noqa: F401
print('imports OK')
"
```
Expected stdout: `imports OK`.

- [ ] **Step 3: Train the smoke test end-to-end**

Run:
```bash
venv/bin/python run.py experiments/smoke_test.py --retrain
```
Expected: progress bar runs to 1000 epochs, then a JSON metrics blob is printed. No NaN lines. No Python tracebacks.

- [ ] **Step 4: Check the generated artifacts and quality floor**

Run:
```bash
venv/bin/python -c "
import json
from pathlib import Path
root = Path('outputs/smoke_test/smoke_test')
for rel in ['checkpoint.msgpack', 'loss.npy', 'metrics.json',
            'plots/loss.png', 'plots/slice.png',
            'plots/pstar_hist2d.png', 'plots/corner_error.png',
            'plots/corner_pstar.png']:
    p = root / rel
    assert p.is_file(), f'missing: {p}'
metrics = json.loads((root / 'metrics.json').read_text())
assert metrics['any_nan_nn'] == 'false', metrics
assert metrics['any_neg_nn'] == 'false', metrics
assert metrics['p95_abs_fstar'] < 1e-1, metrics
print('end-to-end OK:', metrics['p95_abs_fstar'])
"
```
Expected stdout: `end-to-end OK: <a float below 0.1>`.

If `p95_abs_fstar >= 1e-1`, do not mark the task complete. Diagnose: the most common causes are (a) domain clamp too loose at `du = 0.9` letting near-vacuum samples through — tighten to `du_range=(-3.0, 0.7)` temporarily to confirm; (b) `cosine_decay_schedule` peak LR too high for this seed. Flag the observation and loop back to the user before committing any tuning change.

- [ ] **Step 5: Stage nothing — this task is pure verification**

Run: `git status`
Expected: "nothing to commit" aside from `outputs/` (gitignored) and `venv/` (gitignored). If any tracked file is modified here, surface it — something leaked out of earlier tasks.

---

## Self-Review

Checking the plan against the spec section by section:

1. **Spec §Representation** → Task 1 sets `GAS_STATE_DIM = 3`, writes the docstring with the exact definitions of drho/dp/du, documents the `p_ref=1, rho_ref=1` convention. ✓
2. **Spec §physics.py** → Task 1 covers every Keep/Rewrite/Delete item. `two_rarefaction_p0` rewrite matches spec code verbatim. ✓
3. **Spec §model.py** → Task 2 writes the exact `PressureMLP` shown in the spec and deletes the old 5D classes. ✓
4. **Spec §train.py** → Task 3 covers samplers (`uniform`, `r2_quasirandom`) with the new kwargs, the three kept losses (residual, residual_newton, supervised), the dropped `residual_loss_normalized`, the domain key rename, and `evaluate_holdout`. ✓
5. **Spec §plot.py** → Task 4 covers the new `plot_slice` with L↔R swap overlay, 3-pair `_corner_panels`, and relabeled corner plots. ✓
6. **Spec §run.py** → Task 5 updates the lone call site. ✓
7. **Spec §Experiments and top-level scripts** → Task 6 moves every live `experiments/*.py` into `archive/`, deletes `experiments_hipres/`, `check_vacuum.py`, `test_prime.py`. Task 7 writes the fresh `experiments/smoke_test.py`. ✓
8. **Spec §CLAUDE.md** → Task 8 rewrites both sections. ✓
9. **Spec §Verification** →
   - "imports work" → Task 9 Step 2. ✓
   - "grep returns no matches" → Task 9 Step 1 (also partial coverage in Task 6 Step 5). ✓
   - "`run.py experiments/smoke_test.py --retrain` produces all artifacts" → Task 9 Step 3 + Step 4. ✓
   - "`p95_abs_fstar < 1e-1`" → Task 9 Step 4 (with a diagnostic branch if it fails). ✓

**Placeholder scan:** no "TBD", "TODO", "implement later", "similar to Task N", or narrative-only steps. Every code step shows the code. Every verification step shows the command and the expected output.

**Type consistency:** the samplers are called `uniform` everywhere (Task 3 definition, Task 7 import, Task 4 plot.py import, Task 8 docs). `PressureMLP` is consistent between Task 2 (definition) and Task 7 (import). `supervised_loss` is consistent between Task 3 (definition) and Task 7 (import). Domain dict keys `drho_range`/`dp_range`/`du_range` are consistent across Task 3 (sampler kwargs, evaluate_holdout), Task 4 (plot kwargs), Task 5 (`run.py` call), Task 7 (experiment dict), Task 8 (CLAUDE.md doc). `GAS_STATE_DIM = 3` is consistent everywhere it appears.
