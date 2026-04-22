# Simplify and Refactor `riemann_pinn`

**Date:** 2026-04-22
**Status:** Design — awaiting implementation plan

## Goal

Shrink the codebase by removing abstractions that cost more than they earn.
Keep every capability the user currently relies on, except the
primary/correction (stacked-network) machinery, which is dropped for now.

After the refactor, `run.py <experiment_file.py>` loads a `list` of
`Experiment` objects and trains each one. No single-vs-list dual code path,
no string registries, no phase factories, no primary-network plumbing.

## Non-goals

- Changing the physics, the solver, or `plot.py`.
- Changing checkpoint formats beyond what the shape changes force (old
  checkpoints become unloadable; retrain).
- Introducing a config-file format (TOML/YAML). Experiments stay as Python
  files for full flexibility.

## Target module layout

```
riemann_pinn/
    physics.py      # unchanged (~196 lines)
    model.py        # consolidated, ~60 lines
    train.py        # sampler + losses + Experiment/Phase + loop + ckpt + eval (~230 lines)
    plot.py         # unchanged (~312 lines)
run.py              # train CLI (~80 lines)
report.py           # metrics CLI (~50 lines, new)
```

Deleted: `riemann_pinn/experiment.py`, `riemann_pinn/loader.py`,
`run_experiments.sh` (stale).

Approximate total: **~950 lines**, down from 1,465.

## Core data model

In `riemann_pinn/train.py`:

```python
@dataclass
class Phase:
    tx: optax.GradientTransformation
    n_epochs: int
    loss: Callable                    # (params, apply_fn, x) -> (loss, metrics)
    batch_size: int = 2048
    sampler: Callable = uniform_log   # (key, batch_size, **domain) -> x
    fixed_batch: bool = False         # True means don't resample per step (L-BFGS)
    is_lbfgs: bool = False            # picks the L-BFGS call convention in the step fn
    log_every: int = 200
    name: str = "phase"

@dataclass
class Experiment:
    name: str                         # required; goes into outputs/<file_stem>/<name>/
    model: nn.Module
    domain: dict                      # {log_rho_range, log_p_range, u_range}
    phases: list[Phase]
    seed: int = 42
    train_domain: dict | None = None  # optional widening during training
    corner_every: int = 100
```

### What disappears

- **`LOSS_FNS`, `STEP_BUILDERS`, `SAMPLERS` registries.** `loss` and `sampler`
  are always callables; the step function is chosen by `is_lbfgs`.
- **Phase factories** (`adam_cosine`, `adam`, `lbfgs`). Users build their
  `optax.GradientTransformation` inline.
- **`PrimarySpec`, `primary=` field, `load_primary`, `_build_corrected_apply_fn`,
  `_init_correction_state`.** Primary/correction is dropped.
- **`create_train_state_from_apply`** (only primary used it).
- **`Phase.step_kind`, `Phase.split_key_every`.** Replaced by `is_lbfgs` and
  `fixed_batch`.
- **Single-vs-list dual-path.** Experiment files always define
  `experiments = [Experiment(...), ...]` — a one-element list for a single run.

### Example experiment file under the new API

```python
# experiments/geom_w64.py
import optax
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss

DOMAIN = dict(log_rho_range=(0.0, 2.0), log_p_range=(0.0, 2.0), u_range=(-1.0, 1.0))

experiments = [
    Experiment(
        name="geom_w64",
        model=PressureMLP(normalize="geom", width=64, depth=2),
        domain=DOMAIN,
        phases=[
            Phase(
                tx=optax.adamw(optax.cosine_decay_schedule(8e-3, 20_000, 0.0),
                               weight_decay=1e-4),
                n_epochs=20_000,
                loss=residual_loss,
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=1_000,
                loss=residual_loss,
                fixed_batch=True,
                is_lbfgs=True,
            ),
        ],
    ),
]
```

## Models

In `riemann_pinn/model.py`:

```python
class _MLP(nn.Module): ...             # shared MLP block

class PressureMLP(nn.Module):
    """Maps log-space gas state (B, 5) -> scalar p*.

    normalize:
      "none"   — raw log inputs, emits log10 p* directly
      "arith"  — divide by arithmetic mean of (rhoL,rhoR) and (pL,pR);
                 u_ref = c_s(p_ref, rho_ref)
      "geom"   — geometric mean (arithmetic mean in log space),
                 antisymmetric under L<->R
    """
    width: int = 64
    depth: int = 2
    normalize: str = "none"            # "none" | "arith" | "geom"
    activation: Callable = nn.silu
```

### Migration from old classes

- `StarPressureMLP(...)`                  → `PressureMLP(normalize="none", ...)`
- `StarPressureMLPNormalized(...)`        → `PressureMLP(normalize="arith", ...)`
- `StarPressureMLPNormalizedGeom(...)`    → `PressureMLP(normalize="geom", ...)`
- `PstarLogCorrectionMLP`                 → deleted (primary/correction out)
- `StarPressureDS`                        → deleted (no longer needed)
- `StarPressureMLP`'s `output_dim` param  → dropped (no live experiment uses >1)

## Losses

Plain callables in `riemann_pinn/train.py`, all with signature
`(params, apply_fn, x) -> (loss, metrics)`:

- `residual_loss`              — mean `f(p*)^2`
- `residual_loss_normalized`   — mean `(f/c_ref)^2`
- `residual_loss_newton`       — mean `(f/f')^2` proxy for pressure error
- `supervised_loss`            — mean `(p* - p*_true)^2`  (renamed from `residual_loss_supervised`)

Custom losses live in the experiment file that needs them (e.g. the L6 variant
in `adamw_normmlp_geom_widen_L6.py`) and get passed via the `loss=` field.

Dropped: `residual_loss_allfstar` (unused).

## Samplers

- `uniform_log(key, batch_size, *, log_rho_range, log_p_range, u_range)`
- `r2_quasirandom(key, batch_size, *, log_rho_range, log_p_range, u_range)`
  (renamed from `R2_quasirandom` for PEP-8 consistency)

## Training primitives

In `riemann_pinn/train.py`:

```python
def create_train_state(rng, model, tx, batch_size_hint=256): ...

def make_train_step(loss_fn, is_lbfgs=False):
    """JITted step. Branches on is_lbfgs for the L-BFGS call convention."""

def run_phase(state, phase, rng, domain, corner_callback=None, step_offset=0):
    """Run one Phase: build step, loop for n_epochs, return (state, loss_trace)."""

def run_experiment(exp, corner_callback=None):
    """Run all phases; rebuild opt_state at each phase boundary; carry params."""
```

Dropped: `make_lbfgs_train_step` (folded into `make_train_step`),
`create_train_state_from_apply` (primary-only).

Kept unchanged: `save_checkpoint`, `load_checkpoint`, `save_loss_trace`,
`load_loss_trace`, `evaluate_holdout`.

## CLI

### `run.py`

```
run.py <file>                        # train every experiment, sequentially
run.py <file> --index N              # train only experiments[N]
run.py <file> --retrain              # ignore existing checkpoint
run.py <file> --skip-plots
run.py <file> --plot-corner-trace
run.py <file> --count                # print len(experiments) and exit
```

- No `--print-metrics` flag (moved to `report.py`).
- Output path is always `outputs/<file_stem>/<exp.name>/` (no more single-file
  case that omits the name directory).

### `report.py` (new)

```
report.py <file>
```

Reads `outputs/<file_stem>/<exp.name>/metrics.json` for each experiment in the
file and prints a formatted table. Contains the `_METRIC_LABELS`, `_fmt_cell`,
`_print_table`, and `print_metrics` machinery that currently lives in
`run.py`.

### `run_grid.sh`

Minimal change — last line becomes:

```bash
venv/bin/python report.py "$EXPERIMENT"
```

`--count` keeps working.

## Experiments

### Archive (move to `experiments/archive/`)

- `adamw_normmlp_geom_correction.py` — uses primary/correction.
- `lbfgs_normmlp_geom_correction.py` — same.
- `ds_supervised.py` — uses `StarPressureDS`.

### Rewrite in-place (14 files)

Mechanical rewrite per file:

1. Replace `StarPressureMLP`/`StarPressureMLPNormalized`/`StarPressureMLPNormalizedGeom`
   constructor with `PressureMLP(normalize=..., ...)`.
2. Replace `adam_cosine(...)` / `adam(...)` / `lbfgs(...)` factory calls with
   explicit `Phase(tx=optax.chain(...), n_epochs=..., loss=...)`.
3. Replace `loss="fstar"` etc. with imported callables.
4. Convert `experiment = Experiment(...)` singletons into a single-element
   `experiments = [Experiment(name=..., ...)]` list. `name` is required under
   the new API.
5. Delete `sampler="r2"` strings in favor of `sampler=r2_quasirandom`.

Affected: `adam_then_lbfgs.py`, `adam_then_lbfgs_normloss.py`,
`adam_then_lbfgs_normmlp.py`, `adam_then_lbfgs_normmlp_100k.py`,
`adam_then_lbfgs_normmlp_newton.py`, `adamw.py`, `adamw_normmlp.py`,
`adamw_normmlp_geom_widen.py`, `adamw_normmlp_geom_widen_L6.py`,
`adamw_normmlp_geommean.py`, `adamw_normmlp_wdgrid.py`, `lbfgs_ladder.py`,
`lbfgs_small.py`, `smoke_test.py`.

The existing `experiments/archive/*.py` stay untouched — kept only as
reference, unloadable under the new API.

## Breaking changes / migration

- **Checkpoints in `outputs/` become unloadable.** `TrainState` shape depends
  on optimizer structure, and some phases change. Outputs are gitignored;
  mitigation is "retrain". Call this out in the CLAUDE.md update.
- **Old experiment files that are not in the active 14 (and not archived) do
  not work.** No shim for the string registries or factories.
- **`experiment = Experiment(...)` no longer supported**; must be
  `experiments = [Experiment(name=..., ...)]`.
- **`run.py` output path changes** for previously single-valued experiment
  files: was `outputs/<stem>/`, becomes `outputs/<stem>/<name>/`.

## Validation

1. After each commit, run `experiments/smoke_test.py` end-to-end; confirm it
   produces metrics and plots.
2. Spot-check `experiments/adamw_normmlp_geom_widen.py` (a list experiment,
   the widen case) produces metrics in the same ballpark as before the
   refactor. Record pre-refactor `median_abs_fstar` from the current
   `metrics.json` before deleting; compare after retraining.
3. Verify `run_grid.sh experiments/adamw_normmlp_wdgrid.py 2` still runs and
   ends with a metrics table via `report.py`.

## Implementation order (for the plan)

1. New `riemann_pinn/model.py` (`PressureMLP`, `_MLP`) — self-contained.
2. New `riemann_pinn/train.py` (sampler, losses, `Experiment`, `Phase`,
   runner, ckpt, eval). Delete `experiment.py` and `loader.py` in this step.
3. New `run.py` + new `report.py`.
4. Rewrite the 14 active experiments.
5. Archive the 3 dropped experiments. Delete `run_experiments.sh`.
6. Smoke test; fix anything that surfaces.
7. Update `CLAUDE.md` Architecture section to reflect the new layout.
