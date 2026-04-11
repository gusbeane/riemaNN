# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

riemaNN explores solving the 1D Euler Riemann problem with physics-informed neural networks (PINNs). Current focus: predicting the star-region pressure `p*` from left/right gas states by minimizing the squared residual of the exact Riemann jump function `f(p*)`. The codebase is Python 3.12 using JAX + Flax + optax as the compute/training stack. Gas law: ideal gas, `gamma = 5/3`.

## Environment Setup

```bash
source venv/bin/activate
```

The local `venv/` directory contains the Python 3.12 virtual environment. Scripts below use `venv/bin/python` directly so they do not require activation.

## Running Code

Experiments and the compare script are run as Python modules from the repo root so that the `riemann_pinn` package resolves correctly.

- **Train (or re-evaluate) a single experiment:**
  ```bash
  venv/bin/python -m experiments.baseline_w64_d2           # reuse checkpoint if present
  venv/bin/python -m experiments.baseline_w64_d2 --retrain # ignore checkpoint, train from scratch
  ```
  All artifacts are written under `outputs/<experiment_name>/`:
  - `config.json` — resolved config snapshot (for reload)
  - `checkpoint.msgpack` — Flax training state
  - `loss.npy` — per-epoch loss trace
  - `metrics.json` — holdout metrics
  - `plots/loss.png`, `plots/slice.png`

- **Compare multiple experiments:**
  ```bash
  venv/bin/python -m scripts.compare \
      baseline_w64_d2 baseline_w128_d2 baseline_w256_d2 baseline_w512_d2 \
      --label-by model.width \
      --out outputs/_compare/baselines_by_width
  ```
  `--label-by` takes a dotted config key path (e.g. `model.width`, `optimizer.learning_rate`) and uses that value as the plot label. Defaults to the experiment name.

- **Create a new experiment:** copy `experiments/_template.py` to `experiments/<name>.py` and edit. Each experiment is a ~30-line script that instantiates one `Experiment` and calls `.run()`.

- **Notebooks:** `0-damp_osc.ipynb` (PINN for the damped harmonic oscillator) and `1-ideal_riemann.ipynb` (exploratory Riemann-star notebook) are maintained as-is for ad-hoc exploration — they do not share code with `riemann_pinn/`.

There is no test suite, linter, or build system configured.

## Architecture

### `riemann_pinn/` — library package

Dependency order: `paths, physics → models, targets, samplers → losses → training, evaluation, plotting → experiment`.

- **`physics.py`** — `GAMMA = 5/3`, `fjump`, `fstar`, `find_pstar` (Newton iteration), `gas_log_to_phys`, `gas_phys_to_log`, `find_ustar`. All pure JAX.
- **`models.py`** — `StarPressureMLP` (Flax, configurable `width` / `depth` / `activation` / `output_dim`), `MODEL_REGISTRY`, `build_model(spec)`, `model_spec(model)`. Register new model classes in `MODEL_REGISTRY` only if you want compare-script round-tripping; direct imports in experiment scripts work without registration.
- **`targets.py`** — `Target` frozen dataclass with `decode` and `residuals` fields. `STAR_PRESSURE_LOG10` ships with the package (decode: `10**raw`, residuals: `{"fstar": vmap(fstar)}`). Adding a new target (e.g. predict `(p*, u*)`) is ~10 lines and does not require touching the training loop.
- **`losses.py`** — `residual_loss(target, preds, gas_states_log, gas_states_phys, *, weights=None)` iterates over the residual dict returned by the target and returns `(total_loss, per-component metrics dict)`. `supervised_loss` and `combined_loss` are `NotImplementedError` stubs. `make_loss_fn(target, loss_impl, **kwargs)` wraps these into a closure for `jax.value_and_grad`.
- **`samplers.py`** — `uniform_log(rng, batch_size, *, log_rho_range, log_p_range, u_range)`. Contract: `(rng, batch_size) -> (B, 5) array`. The training loop splits rngs; samplers don't thread them.
- **`training.py`** — `build_optimizer(spec)`, `create_train_state`, `make_train_step`, `run_training_loop`, `save_checkpoint`, `load_checkpoint`, `save_loss_trace`, `load_loss_trace`. No "skip if checkpoint exists" logic here — that lives in `Experiment`.
- **`evaluation.py`** — `evaluate_holdout(state, target, n_samples, seed)` is target-generic (percentiles for every residual + pressure-error metrics when the target exposes `pstar`). `slice_grid_data` / `compute_slice_fields` produce 2D heatmap data.
- **`plotting.py`** — `plot_loss`, `plot_slice`, `plot_compare_loss`, `plot_compare_slice`. Forces `matplotlib.use("Agg")` at import so scripts work headless.
- **`experiment.py`** — `Experiment` mutable dataclass with `.run(*, force_retrain, skip_plots)`, `.load(name)`, `.describe()`. Orchestrates the write-config → train-or-load → evaluate → plot cycle and hosts the `extra_evaluators` / `extra_plots` hooks for ad-hoc additions.
- **`paths.py`** — single source of truth for the `outputs/<name>/` layout.

### `experiments/` — runnable experiment scripts

Each file imports from `riemann_pinn`, instantiates one `Experiment`, and exposes a `__main__` guard that calls `exp.run()`. `_template.py` is the starting point; the four `baseline_w*_d2.py` scripts reproduce the widths the previous monolithic script swept via `--compare`.

### `scripts/compare.py` — multi-experiment comparison

Loads any set of experiments by name via `Experiment.load`, extracts labels from a dotted config key path, and writes `loss.png` + `slice.png` side-by-side. Comparison labels are free-form strings, so you can sweep any axis (width, depth, activation, learning rate, loss, sampler).

### Notebooks

`0-damp_osc.ipynb` and `1-ideal_riemann.ipynb` are exploratory and do not share code with `riemann_pinn/`. Leave them alone unless explicitly asked.

## Key Conventions

- JAX 64-bit mode must be enabled before any JAX import path that uses `find_pstar` or large dynamic ranges. Experiment scripts set `jax.config.update("jax_enable_x64", True)` at the top before importing `riemann_pinn`.
- Gas states are always length-5: `(rhoL, pL, rhoR, pR, uRL)` in physical space, or `(log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)` in log space. The constant `GAS_STATE_DIM = 5` lives in `physics.py`.
- Network input is always the log-space representation; the training domain is `log_rho ∈ [-2, 2]`, `log_p ∈ [-2, 2]`, `uRL ∈ [-1, 1]`.
- Per-experiment artifacts (`*.png`, `*.msgpack`, `*.npy`, `outputs/*/config.json`, `outputs/*/metrics.json`, `outputs/*/plots/`) are gitignored. The experiment script under `experiments/<name>.py` is the tracked source of truth; regenerate artifacts with `--retrain` when needed.
- Default random seed is `42`, set per-experiment via the `seed` field. If you want multiple seed replicas, create them as separate experiment scripts (e.g. `baseline_w64_d2_s1`, `baseline_w64_d2_s2`) rather than looping inside one.

## Adding a new experiment

1. Copy `experiments/_template.py` to `experiments/<my_name>.py`.
2. Rename the `name=` field to match the filename.
3. Change whatever you want to vary — model constructor args, `target`, `sampler`, `loss_impl`, `loss_kwargs`, `optimizer`, `n_epochs`, etc.
4. Run: `venv/bin/python -m experiments.<my_name> --retrain`.
5. Compare to baselines: `venv/bin/python -m scripts.compare <my_name> baseline_w64_d2 --label-by model.width`.

## Adding a new abstraction

- **New model architecture**: add a Flax module in `riemann_pinn/models.py`. Register in `MODEL_REGISTRY` only if you need `Experiment.load` to reconstruct it from `config.json` for the compare script.
- **New prediction target** (e.g. predict `(p*, u*)`): add a `Target` in `riemann_pinn/targets.py`. Register in `TARGET_REGISTRY` for the compare script.
- **New loss**: add a function in `riemann_pinn/losses.py` with the signature `(target, preds, gas_states_log, gas_states_phys, **kwargs) -> (loss, metrics)`.
- **New sampler**: add a function in `riemann_pinn/samplers.py` with the signature `(rng, batch_size, **kwargs) -> (B, 5)`.
- **New slice/heatmap quantity**: extend `compute_slice_fields` in `riemann_pinn/evaluation.py` to return additional keys in the returned dict; the plotting functions detect the keys they know how to render.
