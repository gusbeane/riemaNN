# CLAUDE.md

## Project Overview

riemaNN trains a physics-informed neural network (PINN) to predict the star-region pressure `p*/p_ref` from a 3D delta gas state by minimizing the squared residual of the exact Riemann jump function `f(p*)`. Python 3.12, JAX + Flax + optax. Ideal gas, `gamma = 5/3`.

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

Outputs go to `outputs/<file_stem>/<exp.name>/`:
- `metrics.json` — whole-pipeline holdout metrics
- `plots/` — combined-pipeline slice / hist2d / corner plots
- `<stage.name>/checkpoint.msgpack`, `loss.npy`, `metrics.json`, `plots/loss.png` — per stage

Note: if `jax-metal` crashes, set `JAX_PLATFORMS=cpu` before running.

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
- Losses have the signature `(params, apply_fn, gas_states, targets) -> scalar`. The runner derives `targets` outside the loss via `Stage.make_targets(prev_running, pstar_true)` and threads them in. The default loss is `mse_loss`. Stage-level chaining is controlled by `Stage.make_targets` and `Stage.combine` (defaults: multiplicative residual / running product).
- Checkpoints are saved with the last phase's optimizer shape; changing the tail phase's `tx` type invalidates existing checkpoints.
- `experiments/archive/` holds frozen 5D-era experiments for historical reference; they are not expected to import or run against the current code.

## Notebooks

`note/0-damp_osc.ipynb` and `note/1-ideal_riemann.ipynb` are standalone explorations — they do not share code with `riemann_pinn/`.
