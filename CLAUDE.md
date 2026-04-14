# CLAUDE.md

## Project Overview

riemaNN trains a physics-informed neural network (PINN) to predict the star-region pressure `p*` from left/right gas states by minimizing the squared residual of the exact Riemann jump function `f(p*)`. Python 3.12, JAX + Flax + optax. Ideal gas, `gamma = 5/3`.

## Running

```bash
venv/bin/python run.py              # train (or load checkpoint) + evaluate + plot
venv/bin/python run.py --retrain    # ignore checkpoint, train from scratch
venv/bin/python run.py --skip-plots # skip plot generation
```

Outputs go to `outputs/al_w256_d3/`: `checkpoint.msgpack`, `loss.npy`, `metrics.json`, `plots/loss.png`, `plots/slice.png`.

Note: if `jax-metal` crashes, set `JAX_PLATFORMS=cpu` before running.

## Architecture

```
riemann_pinn/
    physics.py   — Riemann solver: GAMMA, fjump, fstar, dfstar_dp, find_pstar, gas_log_to_phys
    model.py     — StarPressureMLP (Flax, configurable width/depth/activation)
    train.py     — uniform_log sampler, residual_loss, build_optimizer, train step/loop,
                   checkpoint I/O, evaluate_holdout
    plot.py      — plot_loss (loss curve), plot_slice (2D heatmap)
run.py           — entry point: Adam (10k) -> L-BFGS (1k), w256 d3
```

## Key Conventions

- `jax.config.update("jax_enable_x64", True)` must be set before JAX ops (done in `run.py`).
- Gas states are length-5: `(rhoL, pL, rhoR, pR, uRL)` in physical space, or `(log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)` in log space. `GAS_STATE_DIM = 5` lives in `physics.py`.
- Network input is log-space; training domain is `log_rho in [-2, 2]`, `log_p in [-2, 2]`, `uRL in [-1, 1]`.
- Output artifacts are gitignored. `run.py` is the source of truth.

## Notebooks

`note/0-damp_osc.ipynb` and `note/1-ideal_riemann.ipynb` are standalone explorations — they do not share code with `riemann_pinn/`.
