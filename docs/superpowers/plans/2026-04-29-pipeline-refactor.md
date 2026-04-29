# Multi-stage Pipeline Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bolted-on `prev_stages` / `evaluate_all_stages` pattern with a first-class `Stage` abstraction inside `Experiment`, support per-stage checkpoint caching, and clear out the agreed dead code (L-BFGS, `train_domain`, `corner_callback`, `output_transform` strings, `supervised_loss`).

**Architecture:** `Experiment` owns an ordered `list[Stage]`. Each `Stage` owns a model and a list of training `Phase`s plus pluggable `make_targets` / `combine` callables (multiplicative-residual defaults). The runner trains each stage in turn, threading the prior stages' running prediction through `make_targets` so stage 0 falls out as supervised regression. Per-stage checkpoints under `outputs/<file>/<exp>/<stage>/`.

**Tech Stack:** JAX + Flax + optax. Python 3.12. No test framework — verification by running `experiments/smoke_test.py` and `experiments/gold_stage2.py` end-to-end and inspecting metrics.

**Spec:** `docs/superpowers/specs/2026-04-29-pipeline-design.md`.

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `riemann_pinn/data.py` | **new** | `Sampler`, `UniformSampler`, `R2QuasirandomSampler`, `DataSet` |
| `riemann_pinn/model.py` | edit | `PressureMLP` with callable `output_transform` (no strings) |
| `riemann_pinn/physics.py` | unchanged | Riemann primitives |
| `riemann_pinn/plot.py` | edit | Plots accept a `predict` callable instead of an `Experiment` |
| `riemann_pinn/train.py` | rewrite | `Experiment`, `Stage`, `Phase`, `mse_loss`, runner, checkpoint I/O, `evaluate_holdout`, `predict_pipeline` |
| `riemann_pinn/__init__.py` | edit | Add `data` to module list |
| `run.py` | edit | Per-stage caching, `--retrain-from`, drop `--plot-corner-trace` |
| `report.py` | edit | Read whole-pipeline `metrics.json` (path unchanged for the top-level file) |
| `plot_losses.py` | edit | Concatenate per-stage `loss.npy` files in stage order |
| `experiments/gold.py` | edit | 1-stage Experiment |
| `experiments/gold_stage2.py` | rename + rewrite | from `gold-stage2.py`; 2-stage Experiment |
| `experiments/smoke_test.py` | edit | 1-stage Experiment |
| `experiments/vary_lr.py` | edit | List of 1-stage Experiments; fix broken `uniform` import |
| `CLAUDE.md` | edit | Reflect new architecture, drop stale `residual_loss_newton` reference |

A note on testing: the spec explicitly excludes adding tests. Each task ends with a concrete *manual verification* step (import the module, run a smoke experiment, etc.) instead of a `pytest` invocation.

---

## Task 1: Extract samplers and DataSet into `riemann_pinn/data.py`

**Files:**
- Create: `riemann_pinn/data.py`
- Modify: `riemann_pinn/train.py` (remove sampler classes and `DataSet`, add re-export shim if needed)
- Modify: `riemann_pinn/__init__.py`

- [ ] **Step 1: Create `riemann_pinn/data.py` with the sampler classes and DataSet, lifted verbatim from `train.py`**

```python
"""Input pipeline: samplers that generate gas-state batches, and DataSet
for fixed precomputed pools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr


class Sampler(ABC):
    """Base class for samplers that generate gas-state batches on the fly."""

    def __init__(
        self,
        *,
        drho_range: tuple[float, float],
        dp_range: tuple[float, float],
        du_range: tuple[float, float],
    ):
        self.drho_range = drho_range
        self.dp_range = dp_range
        self.du_range = du_range

    @abstractmethod
    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        """Draw a batch of gas states. Returns (B, 3)."""
        ...


class UniformSampler(Sampler):
    """Uniform i.i.d. samples in (drho, dp, du)."""

    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        keys = jr.split(rng, 3)
        drho = jr.uniform(keys[0], (batch_size,), minval=self.drho_range[0], maxval=self.drho_range[1])
        dp   = jr.uniform(keys[1], (batch_size,), minval=self.dp_range[0],   maxval=self.dp_range[1])
        du   = jr.uniform(keys[2], (batch_size,), minval=self.du_range[0],   maxval=self.du_range[1])
        return jnp.stack([drho, dp, du], axis=-1)


# R2 quasirandom additive recurrence. Golden ratios for d=1..12.
_R2_GOLDEN = jnp.array([
    1.6180339887498949, 1.3247179572447463, 1.2207440846057596,
    1.1673039782614185, 1.1347241384015194, 1.1127756842787053,
    1.0969815577985598, 1.0850702454914507, 1.0757660660868371,
    1.0682971889208415, 1.0621691678642553, 1.0570505752212287,
])


class R2QuasirandomSampler(Sampler):
    """R2 quasirandom samples in (drho, dp, du)."""

    NDIM = 3

    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        g = _R2_GOLDEN[self.NDIM - 1]
        powers = jnp.arange(1, self.NDIM + 1, dtype=jnp.float32)
        a = g ** (-powers)
        x0 = jr.uniform(rng, (self.NDIM,), minval=0.0, maxval=1.0)
        n = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
        out_unit = jnp.mod(x0[None, :] + n * a[None, :], 1.0)
        lo = jnp.array([self.drho_range[0], self.dp_range[0], self.du_range[0]], dtype=out_unit.dtype)
        hi = jnp.array([self.drho_range[1], self.dp_range[1], self.du_range[1]], dtype=out_unit.dtype)
        return lo + (hi - lo) * out_unit


@dataclass
class DataSet:
    """Fixed precomputed pool of (gas_states, targets). The pipeline runner
    pulls gas_states from here but ignores `targets` -- targets are derived
    by the stage's make_targets from prev_running and pstar_true."""
    gas_states: jnp.ndarray   # (N, 3)
    targets:    jnp.ndarray | None  # (N,)
    head_idx:   int = 0

    def __post_init__(self):
        if self.targets is not None:
            assert self.gas_states.shape[0] == self.targets.shape[0], (
                "gas_states and targets must have the same number of samples"
            )

    def draw_batch(self, batch_size: int):
        assert self.head_idx + batch_size <= self.gas_states.shape[0], (
            "not enough samples left in the dataset"
        )
        start = self.head_idx
        end = self.head_idx + batch_size
        self.head_idx = end
        targets = None if self.targets is None else self.targets[start:end]
        return self.gas_states[start:end], targets
```

- [ ] **Step 2: Remove the sampler classes and DataSet from `train.py`**

In `riemann_pinn/train.py`, delete:
- The `from abc import ABC, abstractmethod` and second `from dataclasses import dataclass` imports near the sampler section
- `class Sampler`, `class UniformSampler`, `_R2_GOLDEN`, `class R2QuasirandomSampler`
- `@dataclass class DataSet` and its `__post_init__` / `draw_batch`
- The `# --- samplers ---` and `# --- training sample dataclass ---` section banners

Add at the top of `train.py` (where the other imports are):

```python
from .data import DataSet, Sampler, UniformSampler
```

`R2QuasirandomSampler` is imported in experiment files directly from `riemann_pinn.data`.

- [ ] **Step 3: Update `riemann_pinn/__init__.py`**

```python
"""riemann_pinn: PINNs for the 1D Euler Riemann problem."""

from . import data, model, physics, plot, train
```

- [ ] **Step 4: Verify imports**

Run:
```bash
venv/bin/python -c "from riemann_pinn import data, train; print(data.UniformSampler, data.DataSet, train.UniformSampler)"
```
Expected: prints three class objects with no exception.

- [ ] **Step 5: Commit**

```bash
git add riemann_pinn/data.py riemann_pinn/train.py riemann_pinn/__init__.py
git commit -m "$(cat <<'EOF'
Extract samplers and DataSet into riemann_pinn/data.py

Pure code move. train.py re-exports Sampler/UniformSampler/DataSet so
existing imports from riemann_pinn.train keep working.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drop string-valued `output_transform` in `PressureMLP`

**Files:**
- Modify: `riemann_pinn/model.py`

- [ ] **Step 1: Replace the model file**

Write `riemann_pinn/model.py`:

```python
"""Neural network model that predicts dimensionless p* = p*/p_ref.

Input is a 3D delta-state (drho, dp, du). Output passes through a
configurable transform; the default is 10**y, giving positive output
matched to the ~1-2 order-of-magnitude range of p*/p_ref over the
sampling domain.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


def _pow10(y: jnp.ndarray) -> jnp.ndarray:
    return 10.0 ** y


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
    """Maps (B, 3) delta state -> (B,) with a configurable output transform."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_transform: Callable[[jnp.ndarray], jnp.ndarray] = _pow10

    @nn.compact
    def __call__(self, x):  # x: (B, 3) = (drho, dp, du)
        model = _MLP(
            width=self.width, depth=self.depth,
            activation=self.activation, output_dim=1,
        )
        return self.output_transform(model(x).squeeze(-1))
```

- [ ] **Step 2: Verify model still constructs and runs**

```bash
venv/bin/python -c "
import jax, jax.numpy as jnp
import jax.random as jr
from riemann_pinn.model import PressureMLP
m = PressureMLP(width=8, depth=1)
p = m.init(jr.PRNGKey(0), jnp.ones((2, 3)))
out = m.apply(p, jnp.ones((4, 3)))
print(out.shape, float(out[0]) > 0)
"
```
Expected: `(4,) True` — output is a length-4 vector of positive values.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/model.py
git commit -m "$(cat <<'EOF'
Drop string-valued output_transform in PressureMLP

Replaced the 'pow10'/'tanh' magic strings with a plain callable that
defaults to 10**y. All existing experiments use the default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `train.py` — Experiment, Stage, Phase, pipeline runner

**Files:**
- Rewrite: `riemann_pinn/train.py`

This is the central refactor. The new file replaces the old top-to-bottom.

- [ ] **Step 1: Write the new `train.py`**

```python
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
from .data import DataSet, Sampler, UniformSampler
from .physics import GAS_STATE_DIM


# --- losses ------------------------------------------------------------------


def mse_loss(params, apply_fn, gas_states, targets):
    """Mean squared error between the network output and runner-derived targets."""
    pred = apply_fn({"params": params}, gas_states)
    return jnp.mean((pred - targets) ** 2)


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
    `jr.fold_in`."""
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
```

The final `train.py` should contain exactly these public/private symbols:

- `mse_loss`
- `_multiplicative_make_targets`, `_multiplicative_combine`
- `Phase`, `Stage`, `Experiment`
- `create_train_state`, `_make_step`, `_draw_gas_states`, `run_stage`, `build_template_state`
- `predict_pipeline`
- `save_checkpoint`, `load_checkpoint`, `save_loss_trace`, `load_loss_trace`
- `evaluate_holdout`

…plus re-imports of `Sampler`, `UniformSampler`, `DataSet` from `data.py` (so existing `from riemann_pinn.train import UniformSampler` still works during the transition). No `run_phase`, no `run_experiment`, no `supervised_loss`, no `residual_loss`, no L-BFGS step variant, no `corner_callback` plumbing, no `train_domain`, no `prev_stages`.

- [ ] **Step 2: Verify the module imports and the dataclasses construct**

```bash
venv/bin/python -c "
import optax
from riemann_pinn.train import Experiment, Stage, Phase, mse_loss, predict_pipeline
from riemann_pinn.model import PressureMLP
from riemann_pinn.data import UniformSampler

dom = dict(drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.5))
exp = Experiment(
    name='x', seed=42, domain=dom,
    stages=[Stage(name='main', model=PressureMLP(width=4, depth=1),
                  phases=[Phase(tx=optax.adam(1e-3), n_epochs=2,
                                loss=mse_loss, batch_size=8,
                                sampler=UniformSampler(**dom))])],
)
print(len(exp.stages), exp.stages[0].name, exp.stages[0].make_targets.__name__, exp.stages[0].combine.__name__)
"
```
Expected: `1 main _multiplicative_make_targets _multiplicative_combine`.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/train.py
git commit -m "$(cat <<'EOF'
Refactor train.py: Stage abstraction, pipeline runner

Experiment now owns list[Stage]; each Stage owns list[Phase] plus
make_targets / combine callables (multiplicative defaults). The runner
threads prev_running through stages so stage 0 falls out as supervised
regression. Drops L-BFGS, train_domain, corner_callback, supervised_loss,
prev_stages, evaluate_all_stages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update `plot.py` to use `predict_pipeline`

**Files:**
- Modify: `riemann_pinn/plot.py`

The plots currently call `exp.evaluate_all_stages(gas_states)`, which is gone. They now take a `predict` callable instead of an `Experiment`.

- [ ] **Step 1: Replace `plot.py`**

The full new file:

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
from .data import UniformSampler  # noqa: E402


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
    predict, out_path: Path, *, n: int = 250,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9),
    du_slice: float = 0.0,
    err_range=(-0.1, 0.1), nbins: int = 100, name: str | None = None,
) -> None:
    """Three-panel slice over (drho, dp) at du = du_slice. `predict` maps
    a (B, 3) gas-state batch to a (B,) pstar prediction."""
    dr = jnp.linspace(*drho_range, n)
    dpv = jnp.linspace(*dp_range, n)
    dr_grid, dp_grid = jnp.meshgrid(dr, dpv, indexing="ij")
    gas_states = jnp.stack([
        dr_grid.ravel(), dp_grid.ravel(),
        jnp.full(n * n, du_slice),
    ], axis=-1)

    pstar_nn = predict(gas_states)
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

    swap_sign = jnp.array([-1.0, -1.0, 1.0])
    gas_states_swap = gas_states * swap_sign
    pstar_nn_swap = predict(gas_states_swap)
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


# --- corner plots ------------------------------------------------------------


def _corner_panels(n, *, drho_range, dp_range, du_range):
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
    predict, out_path: Path, *, n: int = 50,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_NN / p*_true) over all 3 input-variable pairs."""
    panels, axes_np = _corner_panels(
        n, drho_range=drho_range, dp_range=dp_range, du_range=du_range,
    )
    all_gas = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_nn_all = predict(all_gas)
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
    predict, out_path: Path, *, n_samples: int = 50_000, seed: int = 999,
    nbins: int = 200, name: str | None = None,
    **domain_kwargs,
) -> None:
    """2D histogram of log10(p*_true) vs log10(p*_NN)."""
    import jax.random as jr

    rng = jr.PRNGKey(seed)
    sampler = UniformSampler(**domain_kwargs)
    gas_states = sampler.draw_batch(rng, n_samples)

    pstar_nn = predict(gas_states)
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

The signatures changed: `plot_slice`, `plot_corner_error`, `plot_pstar_hist2d` now take `predict` (a callable) as the first positional argument instead of `exp`. Callers in `run.py` get updated in Task 5.

- [ ] **Step 2: Verify import**

```bash
venv/bin/python -c "
from riemann_pinn import plot
import inspect
print(list(inspect.signature(plot.plot_slice).parameters)[:2])
"
```
Expected: `['predict', 'out_path']`.

- [ ] **Step 3: Commit**

```bash
git add riemann_pinn/plot.py
git commit -m "$(cat <<'EOF'
Plot functions take a predict callable, not an Experiment

evaluate_all_stages is gone; plots now accept a (gas_states) -> pstar
callable that the caller builds via predict_pipeline. Decouples the
plotting layer from the experiment dataclass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update `run.py` for per-stage caching and `--retrain-from`

**Files:**
- Rewrite: `run.py`

- [ ] **Step 1: Replace `run.py`**

```python
"""Run a PINN training experiment defined by a Python file.

Experiment files must export `experiments = [Experiment(...), ...]`.
Outputs land in outputs/<file_stem>/<exp.name>/<stage.name>/ for each
stage's checkpoint, loss, and per-stage metrics, plus a top-level
metrics.json and plots/ for the full pipeline.
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
    load_loss_trace, predict_pipeline, run_stage, save_checkpoint,
    save_loss_trace,
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
            raise ValueError(f"{path}: experiments[{i}] must set `name=...`")
        if not e.stages:
            raise ValueError(f"{path}: experiments[{i}] has empty `stages`")
        stage_names = [s.name for s in e.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError(
                f"{path}: experiments[{i}].stages has duplicate names: {stage_names}"
            )
        for j, s in enumerate(e.stages):
            if not s.name:
                raise ValueError(
                    f"{path}: experiments[{i}].stages[{j}] must set `name=...`"
                )
    names = [e.name for e in exps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate names in `experiments`: {names}")
    return list(exps)


def _stage_dir(exp_dir: Path, stage_name: str) -> Path:
    return exp_dir / stage_name


def _wipe_from(exp_dir: Path, stages, retrain_from: str | None) -> None:
    """Delete the checkpoint dir for `retrain_from` and every later stage."""
    if retrain_from is None:
        return
    names = [s.name for s in stages]
    if retrain_from not in names:
        raise SystemExit(
            f"--retrain-from {retrain_from!r} not in this experiment's stages: {names}"
        )
    start = names.index(retrain_from)
    for s in stages[start:]:
        d = _stage_dir(exp_dir, s.name)
        if d.exists():
            print(f"  wiping {d}")
            shutil.rmtree(d)


def _train_pipeline(exp: Experiment, exp_dir: Path) -> tuple[list, list]:
    """Train each stage, skipping those with an existing checkpoint.

    Returns (stage_states, per_stage_traces) where stage_states is a list
    of (Stage, TrainState) and per_stage_traces is a list of np arrays
    aligned with exp.stages."""
    stage_states: list = []
    traces: list = []

    for i, stage in enumerate(exp.stages):
        sdir = _stage_dir(exp_dir, stage.name)
        ckpt_path = sdir / "checkpoint.msgpack"
        loss_path = sdir / "loss.npy"
        if ckpt_path.is_file():
            print(f"[{i+1}/{len(exp.stages)}] {stage.name}: loading checkpoint")
            state = load_checkpoint(ckpt_path, build_template_state(stage))
            trace = load_loss_trace(loss_path)
        else:
            print(f"[{i+1}/{len(exp.stages)}] {stage.name}: training")
            prev_specs = [
                (st.apply_fn, st.params, prev_stage.combine)
                for prev_stage, st in stage_states
            ]
            t0 = time.monotonic()
            state, trace, _ = run_stage(
                stage, prev_specs, exp_seed=exp.seed, stage_index=i,
            )
            elapsed = round(time.monotonic() - t0, 1)
            save_checkpoint(ckpt_path, state)
            save_loss_trace(loss_path, trace)
            (sdir / "stage_train_seconds.txt").write_text(f"{elapsed}\n")

        stage_states.append((stage, state))
        traces.append(trace)
    return stage_states, traces


def _train_and_eval(
    exp: Experiment, exp_path: Path, exp_dir: Path, name: str,
    *, retrain: bool, retrain_from: str | None, skip_plots: bool,
) -> None:
    if retrain and exp_dir.exists():
        print(f"  --retrain: wiping {exp_dir}")
        shutil.rmtree(exp_dir)
    _wipe_from(exp_dir, exp.stages, retrain_from)
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_path, exp_dir / exp_path.name)

    t0 = time.monotonic()
    stage_states, traces = _train_pipeline(exp, exp_dir)
    pipeline_time_s = round(time.monotonic() - t0, 1)

    # Per-stage metrics: evaluate the running pipeline truncated at each stage.
    for k, (stage, _state) in enumerate(stage_states):
        sdir = _stage_dir(exp_dir, stage.name)
        m = evaluate_holdout(stage_states[: k + 1], **exp.domain)
        with (sdir / "metrics.json").open("w") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        if not skip_plots and traces[k] is not None:
            plot_loss(traces[k], sdir / "plots" / "loss.png",
                      title=f"Training loss — {name}/{stage.name}")

    # Whole-pipeline metrics.
    metrics = evaluate_holdout(stage_states, **exp.domain)
    metrics["training_time_s"] = pipeline_time_s
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if skip_plots:
        return
    predict = lambda gs: predict_pipeline(stage_states, gs)
    plots_dir = exp_dir / "plots"
    plot_slice(predict, plots_dir / "slice.png",
               drho_range=exp.domain["drho_range"],
               dp_range=exp.domain["dp_range"], name=name)
    plot_pstar_hist2d(predict, plots_dir / "pstar_hist2d.png", name=name, **exp.domain)
    plot_corner_error(predict, plots_dir / "corner_error.png", name=name, **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", name=name, **exp.domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="train only experiments[N] (default: all)")
    ap.add_argument("--retrain", action="store_true",
                    help="wipe and retrain every stage")
    ap.add_argument("--retrain-from", default=None,
                    help="wipe this stage and all later stages, then retrain")
    ap.add_argument("--skip-plots", action="store_true")
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
        parent = Path(exp.output_root) if exp.output_root else out_root / stem
        exp_dir = parent / exp.name
        name = f"{stem}/{exp.name}"
        _train_and_eval(
            exp, exp_path, exp_dir, name,
            retrain=args.retrain, retrain_from=args.retrain_from,
            skip_plots=args.skip_plots,
        )


if __name__ == "__main__":
    main()
```


- [ ] **Step 2: Verify import**

```bash
venv/bin/python -c "import run; print(run.main is not None)"
```
Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "$(cat <<'EOF'
run.py: per-stage caching and --retrain-from

Each stage gets its own subdirectory with checkpoint, loss, and per-stage
metrics.json. Stages with an existing checkpoint are skipped. New
--retrain-from <stage> wipes that stage and everything after it. Removes
--plot-corner-trace and the corner-trace frames machinery.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update `report.py` and `plot_losses.py` for the new layout

**Files:**
- Modify: `report.py`
- Modify: `plot_losses.py`

`report.py` reads the top-level `<exp>/metrics.json`, which still exists, but the load path needs to handle missing files gracefully (a partially-trained pipeline may have per-stage metrics but no whole-pipeline metrics). `plot_losses.py` needs to concatenate per-stage `loss.npy` in stage order.

- [ ] **Step 1: Update `plot_losses.py`**

Replace the `_loss_path` and trace-loading section with logic that walks the stage directories:

```python
"""Overlay training-loss curves for every experiment in a file.

Reads outputs/<file_stem>/<exp.name>/<stage.name>/loss.npy for each stage
of each experiment, concatenates them in stage order, and writes a single
plot to outputs/<file_stem>/plots/loss_compare.png. Missing per-stage
loss traces are skipped with a warning."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import load_experiments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--out", default=None,
                    help="output path (default: outputs/<stem>/plots/loss_compare.png)")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    def _exp_dir(e):
        parent = Path(e.output_root) if e.output_root else out_root / stem
        return parent / e.name

    traces: list[tuple[str, np.ndarray]] = []
    for e in exps:
        parts = []
        for s in e.stages:
            p = _exp_dir(e) / s.name / "loss.npy"
            if not p.is_file():
                print(f"warning: no loss.npy for {e.name}/{s.name} (expected {p}); skipping experiment")
                parts = None
                break
            parts.append(np.load(p))
        if parts:
            traces.append((e.name, np.concatenate(parts)))

    if not traces:
        raise SystemExit(f"no loss traces found for any experiment in {exp_path}")

    out_path = Path(args.out) if args.out else out_root / stem / "plots" / "loss_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for name, arr in traces:
        ax.plot(np.log10(arr), label=name, linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
    ax.set_ylim(-8, 4)
    ax.set_title(f"Training loss — {stem}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=max(1, len(traces) // 10))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(traces)} curves)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `report.py` is already correct**

The whole-pipeline metrics still live at `<exp>/metrics.json`, which is what `report.py` reads. No code change needed; verify it still imports:

```bash
venv/bin/python -c "import report; print(report.main is not None)"
```
Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add plot_losses.py
git commit -m "$(cat <<'EOF'
plot_losses: concatenate per-stage loss.npy in stage order

report.py is unchanged: top-level metrics.json path is the same.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migrate the experiment files

**Files:**
- Modify: `experiments/gold.py`
- Modify: `experiments/smoke_test.py`
- Modify: `experiments/vary_lr.py`
- Delete: `experiments/gold-stage2.py`
- Create: `experiments/gold_stage2.py`

- [ ] **Step 1: Rewrite `experiments/smoke_test.py`**

```python
"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.9),
)


experiments = [
    Experiment(
        name="smoke_test",
        seed=42,
        domain=_DOMAIN,
        stages=[
            Stage(
                name="main",
                model=PressureMLP(width=64, depth=2),
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
                        loss=mse_loss,
                        batch_size=2048,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=100,
                        name="adam_cosine",
                    ),
                ],
            ),
        ],
    ),
]
```

- [ ] **Step 2: Rewrite `experiments/gold.py`**

```python
"""Gold reference network: single-stage AdamW training in the 3D delta-space."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16
LR = 2e-3


experiments = [
    Experiment(
        name="gold",
        seed=42,
        domain=_DOMAIN,
        stages=[
            Stage(
                name="main",
                model=PressureMLP(width=16, depth=2),
                phases=[
                    Phase(
                        tx=optax.chain(
                            optax.clip_by_global_norm(1.0),
                            optax.adamw(learning_rate=LR),
                        ),
                        n_epochs=N_EPOCHS,
                        loss=mse_loss,
                        batch_size=BATCH_SIZE,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=1,
                        name="adamw",
                    ),
                ],
            ),
        ],
    ),
]
```

- [ ] **Step 3: Rewrite `experiments/vary_lr.py` (also fixes the broken `uniform` import)**

```python
"""Sweep AdamW learning rate on the gold-style architecture."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16


experiments = [
    Experiment(
        name=f"lr{lr}",
        seed=42,
        domain=_DOMAIN,
        stages=[
            Stage(
                name="main",
                model=PressureMLP(width=16, depth=2),
                phases=[
                    Phase(
                        tx=optax.chain(
                            optax.clip_by_global_norm(1.0),
                            optax.adamw(learning_rate=lr),
                        ),
                        n_epochs=N_EPOCHS,
                        loss=mse_loss,
                        batch_size=BATCH_SIZE,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=1,
                        name="adamw",
                    ),
                ],
            ),
        ],
    )
    for lr in [1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3, 8e-3]
]
```

- [ ] **Step 4: Delete `experiments/gold-stage2.py` and create `experiments/gold_stage2.py`**

```bash
git rm experiments/gold-stage2.py
```

Then write `experiments/gold_stage2.py`:

```python
"""Two-stage gold pipeline: stage 0 predicts p* directly, stage 1 predicts
the multiplicative correction p*_true / p*_stage0. Default Stage
make_targets / combine handle the residual chaining."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16
LR = 2e-3


def _stage(name: str) -> Stage:
    return Stage(
        name=name,
        model=PressureMLP(width=16, depth=2),
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(learning_rate=LR),
                ),
                n_epochs=N_EPOCHS,
                loss=mse_loss,
                batch_size=BATCH_SIZE,
                sampler=UniformSampler(**_DOMAIN),
                log_every=1,
                name="adamw",
            ),
        ],
    )


experiments = [
    Experiment(
        name="gold_stage2",
        seed=42,
        domain=_DOMAIN,
        stages=[_stage("base"), _stage("correction")],
    ),
]
```

- [ ] **Step 5: Verify the experiment files load**

```bash
venv/bin/python -c "
from run import load_experiments
from pathlib import Path
for p in [
    'experiments/smoke_test.py',
    'experiments/gold.py',
    'experiments/vary_lr.py',
    'experiments/gold_stage2.py',
]:
    es = load_experiments(Path(p))
    print(p, [(e.name, [s.name for s in e.stages]) for e in es])
"
```
Expected: each line prints the experiment file path and a list of `(exp_name, [stage_names...])`. `gold_stage2.py` shows two stages, the others show one.

- [ ] **Step 6: Commit**

```bash
git add experiments/gold.py experiments/smoke_test.py experiments/vary_lr.py experiments/gold_stage2.py
git rm experiments/gold-stage2.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
Migrate experiment files to Stage API

- gold.py, smoke_test.py, vary_lr.py: 1-stage Experiments.
- gold_stage2.py: 2-stage Experiment using default multiplicative
  composition. Replaces the hyphen-named gold-stage2.py and drops the
  manual checkpoint load + 10M.npz residual derivation.
- vary_lr.py also fixes a broken `uniform` import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Architecture and Conventions sections**

Replace the `## Architecture` block in `CLAUDE.md` with:

```markdown
## Architecture

```
riemann_pinn/
    physics.py   — 3D Riemann primitives: GAMMA, ALPHA, BETA, MU, GAS_STATE_DIM (=3),
                   get_ducrit, ftilde, fstar, dfstar_dp, find_pstar, two_rarefaction_p0
    data.py      — Sampler ABC, UniformSampler, R2QuasirandomSampler, DataSet
    model.py     — _MLP block + PressureMLP (3D input -> p*/p_ref via callable
                   output_transform, default 10**y)
    train.py     — Experiment / Stage / Phase dataclasses, mse_loss,
                   pipeline runner (run_stage), predict_pipeline,
                   checkpoint I/O, evaluate_holdout
    plot.py      — plot_loss, plot_slice, plot_corner_error, plot_corner_pstar,
                   plot_pstar_hist2d (all take a `predict` callable)
run.py           — CLI: load experiments, train each stage (skipping cached
                   checkpoints), save metrics, plot
report.py        — CLI: read whole-pipeline metrics.json files, print table
plot_losses.py   — CLI: concatenate per-stage loss.npy curves and overlay
                   them in outputs/<stem>/plots/loss_compare.png
```

Experiment files export `experiments = [Experiment(...), ...]`. Each
`Experiment` has `name`, `seed`, `domain`, `stages`. Each `Stage` has
`name`, `model`, `phases`, plus optional `make_targets` / `combine`
callables (multiplicative defaults). Each `Phase` has `tx`
(`optax.GradientTransformation`), `n_epochs`, `loss`, `batch_size`,
`sampler`, `log_every`, `name`.

Outputs go to `outputs/<file_stem>/<exp.name>/`:
- `metrics.json` — whole-pipeline holdout metrics
- `plots/` — combined-pipeline slice / hist2d / corner plots
- `<stage.name>/checkpoint.msgpack`, `loss.npy`, `metrics.json`,
  `plots/loss.png` — per stage
```

Replace the `## Key Conventions` block with the same content but with these edits:

- Remove the line about `train_domain` and the `corner_every` mention.
- Replace `Losses have the signature ... Custom losses live in the experiment file that needs them and are passed via Phase(loss=...).` with:

  > Losses have the signature `(params, apply_fn, gas_states, targets) -> scalar`. The runner derives `targets` outside the loss via `Stage.make_targets(prev_running, pstar_true)` and threads them in. The default loss is `mse_loss`. Stage-level chaining is controlled by `Stage.make_targets` and `Stage.combine` (defaults: multiplicative residual / running product).

- Remove the bullet about `experiments/archive/` if the user has cleaned it up; otherwise leave it.

Replace the `## Running` block — the only change is replacing `--retrain` with the new flag set:

```markdown
## Running

```bash
venv/bin/python run.py experiments/smoke_test.py                         # train + evaluate + plot
venv/bin/python run.py experiments/smoke_test.py --retrain               # wipe and retrain every stage
venv/bin/python run.py experiments/smoke_test.py --retrain-from main     # retrain from this stage onward
venv/bin/python run.py experiments/smoke_test.py --skip-plots            # skip plot generation
venv/bin/python run.py experiments/foo.py --index 2                      # list-valued file: train only experiments[2]
venv/bin/python run.py experiments/foo.py --count                        # print number of experiments
venv/bin/python report.py experiments/foo.py                             # print metrics table
venv/bin/python plot_losses.py experiments/foo.py                        # overlay all loss curves
./run_grid.sh experiments/foo.py 4                                       # fan out a list over 4 workers + report + loss compare
```
```

(Remove any reference to `residual_loss_newton` if still present — the symbol does not exist in `train.py`.)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
CLAUDE.md: document Stage abstraction and per-stage layout

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end verification

**Files:** none (just running things).

- [ ] **Step 1: Run the smoke test fresh**

```bash
rm -rf outputs/smoke_test
venv/bin/python run.py experiments/smoke_test.py
```

Expected: training progress bar runs for 1000 epochs, JSON metrics print at the end, `outputs/smoke_test/smoke_test/` contains `metrics.json`, `plots/`, and `main/` subdir with `checkpoint.msgpack`, `loss.npy`, `metrics.json`, `plots/loss.png`. `metrics.json` should show `median_abs_delta_log10_p` somewhere in the 1e-2 to 1e-3 range (matches today's smoke baseline).

- [ ] **Step 2: Re-run the smoke test — should skip training**

```bash
venv/bin/python run.py experiments/smoke_test.py
```

Expected: prints `[1/1] main: loading checkpoint`, no training, finishes within seconds.

- [ ] **Step 3: Run the two-stage pipeline**

```bash
rm -rf outputs/gold_stage2
venv/bin/python run.py experiments/gold_stage2.py
```

Expected: trains stage `base`, then stage `correction`. `outputs/gold_stage2/gold_stage2/` has `metrics.json` (whole pipeline), plus `base/` and `correction/` stage subdirs each with their own checkpoint, loss, plots. The whole-pipeline `metrics.json` prints without `nan`s. Compare `base/metrics.json` against the whole-pipeline `metrics.json`: the user has confirmed that multiplicative correction without `eps` works in practice, so `correction` should match or improve on `base` after training. If `correction` makes things worse, the network's output transform may need tuning (e.g., a transform centered at 1 — `lambda y: 1 + y` or `jnp.exp`) — flag this back to the user rather than fixing it in the refactor.

- [ ] **Step 4: Test `--retrain-from`**

```bash
venv/bin/python run.py experiments/gold_stage2.py --retrain-from correction
```

Expected: prints `wiping outputs/gold_stage2/gold_stage2/correction`, loads `base` checkpoint without retraining, retrains `correction`, finishes.

- [ ] **Step 5: Test `--retrain`**

```bash
venv/bin/python run.py experiments/gold_stage2.py --retrain
```

Expected: wipes the entire `outputs/gold_stage2/gold_stage2` dir, retrains both stages from scratch.

- [ ] **Step 6: Test reporting**

```bash
venv/bin/python report.py experiments/gold_stage2.py
venv/bin/python plot_losses.py experiments/gold_stage2.py
```

Expected: `report.py` prints metric labels and values; `plot_losses.py` writes `outputs/gold_stage2/plots/loss_compare.png` with one curve covering both stages concatenated.

- [ ] **Step 7: Spot-check `vary_lr.py` loads (optional, full run is heavy)**

```bash
venv/bin/python run.py experiments/vary_lr.py --count
```

Expected: `7`.

- [ ] **Step 8: Final commit if any docs / README polish was needed during verification**

If steps 1-7 surfaced anything (e.g., a typo in CLAUDE.md, a filename mismatch), fix and commit. Otherwise this task ends with a clean working tree.

```bash
git status
```
Expected: clean tree, last commit is the CLAUDE.md update from Task 8.
