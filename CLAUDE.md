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
venv/bin/python plot_losses.py experiments/foo.py                # overlay all loss curves
./run_grid.sh experiments/foo.py 4                               # fan out a list over 4 workers + report + loss compare
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
plot_losses.py   — CLI: overlay each experiment's loss.npy into outputs/<stem>/plots/loss_compare.png
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
