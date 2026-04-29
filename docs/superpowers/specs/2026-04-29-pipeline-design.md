# Multi-stage Pipeline Refactor

## Background

`riemaNN` trains a PINN to predict `p*/p_ref` from a 3D delta gas state. The
prototype `experiments/gold-stage2.py` chains a second NN that predicts a
multiplicative correction on top of the gold network's output, and it works
well enough to keep. The chaining today is bolted on through
`Experiment.prev_stages: list[(state, eps)]` and a special `evaluate_all_stages`
method, with the residual dataset and `eps` normalization assembled by hand
inside the experiment file. This refactor turns multi-NN chaining into a
first-class concept and clears out unused code paths along the way.

## Goals

- A `Stage` abstraction nested inside `Experiment`. An experiment owns an
  ordered list of stages; each stage trains one NN and composes with the
  prior stages' running prediction.
- Pluggable composition: each stage carries `make_targets` and `combine`
  callables with multiplicative-residual defaults.
- Per-stage checkpointing so re-running an experiment retrains only the
  stages that need it. `--retrain-from <stage>` for explicit invalidation.
- Drop the unused L-BFGS, `train_domain`, `output_transform`-string,
  `R2QuasirandomSampler`, `corner_callback`, and `eps` paths.

## Non-Goals

- Automatic upstream-change detection for cache invalidation. The user is
  responsible for `--retrain-from` when they tweak an upstream stage.
- Generalizing `make_targets`/`combine` beyond `(prev_running, ...)` inputs.
  `gas_states` can be added later if a use case appears.
- Tests. There are none today and adding them is out of scope.

## Architecture

Three nested dataclasses, each with a single responsibility:

```
Experiment
├── name: str
├── seed: int
├── domain: dict          # drho_range, dp_range, du_range
├── output_root: str | Path | None
└── stages: list[Stage]

Stage
├── name: str
├── model: nn.Module
├── make_targets: Callable[[prev_running, pstar_true], targets]
│       default: lambda prev, true: true / prev
├── combine: Callable[[prev_running, stage_output], new_running]
│       default: lambda prev, out: prev * out
└── phases: list[Phase]

Phase
├── tx: optax.GradientTransformation
├── n_epochs: int
├── loss: Callable[[params, apply_fn, gas_states, targets], scalar]
├── batch_size: int
├── sampler: Sampler | DataSet
├── log_every: int = 200
└── name: str = "phase"
```

`Phase` retains its multi-optimizer-per-NN role (e.g., AdamW followed by a
cosine schedule). `Stage` is new. `Experiment` keeps its name to minimize
import churn.

### Stage 0 unification

Stage 0 is not special-cased. The runner threads
`prev_running = jnp.ones_like(pstar_true)` into the first stage. With the
default `make_targets` and `combine`, this gives:

- `make_targets(1.0, pstar_true) = pstar_true / 1.0 = pstar_true`
- `combine(1.0, output)         = 1.0 * output     = output`

So a single-stage Experiment behaves as supervised regression against
`pstar_true`. The previously-separate `supervised_loss` is therefore subsumed
and removed.

## Data flow

The runner builds a JIT'd train step per stage that closes over the prior
stages' frozen `TrainState`s and their `combine` callables:

```python
def make_step(stage, prev_states_and_combines):
    @jax.jit
    def step(state, rng):
        gas_states = sampler.draw_batch(rng, batch_size)
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
        prev_running = jnp.ones_like(pstar_true)
        for prev_state, prev_combine in prev_states_and_combines:
            out = prev_state.apply_fn({"params": prev_state.params}, gas_states)
            prev_running = prev_combine(prev_running, out)
        targets = stage.make_targets(prev_running, pstar_true)
        loss, grads = jax.value_and_grad(
            lambda p: phase.loss(p, state.apply_fn, gas_states, targets)
        )(state.params)
        return state.apply_gradients(grads=grads), loss
    return step
```

- Prior `TrainState`s travel as JAX pytree leaves; they're recompiled once per
  stage (acceptable: stages are O(1)).
- `pstar_true` per batch via `vmap(find_pstar)` is the same cost the current
  stage 1 already pays.
- The user's `phase.loss` keeps its existing
  `(params, apply_fn, gas_states, targets) -> scalar` signature; the runner
  derives `targets` outside the loss and passes them in.

When `phase.sampler` is a `DataSet`, the runner pulls `(gas_states, _)` from
it; the stored `targets` are ignored in favor of the runner-derived
`make_targets` output. Vestigial but acceptable since `DataSet` is no longer
the primary sampler.

### Inference / evaluation

```python
def predict(states_and_stages, gas_states):
    running = jnp.ones((gas_states.shape[0],))
    for state, stage in states_and_stages:
        out = state.apply_fn({"params": state.params}, gas_states)
        running = stage.combine(running, out)
    return running
```

`evaluate_holdout` switches from a single state to a list of
`(state, stage)` pairs; metrics are otherwise unchanged.

## CLI and output layout

```
outputs/<file_stem>/<exp.name>/
├── <exp_file>.py
├── metrics.json              ← whole-pipeline holdout metrics
├── plots/                    ← combined-pipeline slice, hist2d, corner_*
└── <stage.name>/
    ├── checkpoint.msgpack
    ├── loss.npy
    ├── metrics.json          ← this stage's incremental contribution
    └── plots/
        └── loss.png
```

The experiment source file is copied once into `<exp.name>/`, not per stage.

CLI:

| Flag | Behavior |
|------|----------|
| `run.py exp.py` | For each experiment, train each stage in order, skipping stages whose checkpoint exists. |
| `--retrain` | Wipe and retrain every stage of every selected experiment. |
| `--retrain-from <stage_name>` | Wipe that stage's dir and all later stages' dirs, then retrain. Errors if name doesn't match. |
| `--index N` | Only `experiments[N]`. |
| `--count` | Print `len(experiments)` and exit. |
| `--skip-plots` | Skip plot generation. |

`--plot-corner-trace` is removed.

## Cleanup checklist

Removed:

- `Phase.is_lbfgs`, `Phase.fixed_batch`, the L-BFGS branch in `make_train_step`.
- `Experiment.train_domain`, `Experiment.corner_every`, `Experiment.state`,
  `Experiment.prev_stages`, `Experiment.evaluate_all_stages`.
- `R2QuasirandomSampler`.
- `supervised_loss` (subsumed by default `make_targets`).
- `corner_callback` plumbing in `run_phase` / `run_experiment`,
  `--plot-corner-trace`, `corner_frames/` output directory.
- `eps` normalization machinery from `gold-stage2.py`.
- Magic-string `output_transform` in `PressureMLP`. Replaced with a plain
  callable default `lambda y: 10.0 ** y`.
- Stale `residual_loss_newton` mention in `CLAUDE.md`.

Kept:

- `Phase` (multi-optimizer per NN).
- `DataSet` (secondary sampler path; runner-derived targets override its
  stored targets).
- `evaluate_holdout` and its metrics dict. It now takes a list of
  `(state, stage)` pairs and returns metrics for the running prediction
  through stage `k`. The runner calls it once at the full pipeline
  (`<exp>/metrics.json`) and once per stage truncating to that stage
  (`<exp>/<stage>/metrics.json`).
- Standalone end-of-training corner plots (`plot_corner_error`,
  `plot_corner_pstar`).

## Migration of existing experiment files

- `experiments/gold.py` → 1-stage Experiment, `UniformSampler` +
  `mse_loss` (existing `residual_loss` renamed; it's no longer
  residual-specific now that the runner derives `targets`). Default
  `make_targets`/`combine`, `prev_running=1` reproduces today's
  supervised behavior.
- `experiments/gold-stage2.py` → 2-stage Experiment. Stage 0 is gold;
  stage 1 inherits the default multiplicative composition. Drops the
  10M.npz load, `DataSet` setup, and `eps` plumbing.
- `experiments/smoke_test.py` → 1-stage Experiment.
- `experiments/vary_lr.py` → list of 1-stage Experiments.

The hyphen in `gold-stage2.py` is replaced with an underscore for module
hygiene (`gold_stage2.py`); `run.py` uses `spec_from_file_location` so this
is cosmetic.

`CLAUDE.md` updated to reflect the new architecture and remove the stale
`residual_loss_newton` reference.
