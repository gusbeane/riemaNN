# CLAUDE.md

## Project Overview

riemaNN trains a physics-informed neural network (PINN) to predict the star-region pressure `p*` from left/right gas states by minimizing the squared residual of the exact Riemann jump function `f(p*)`. Python 3.12, JAX + Flax + optax. Ideal gas, `gamma = 5/3`.

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

## Notebooks

`note/0-damp_osc.ipynb` and `note/1-ideal_riemann.ipynb` are standalone explorations — they do not share code with `riemann_pinn/`.
