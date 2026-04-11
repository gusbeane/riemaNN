# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

riemaNN explores solving the Riemann problem (1D Euler equations for ideal gas) using both classical numerical methods and neural networks / PINNs. The codebase is Python 3.12, using JAX as the primary compute backend.

## Environment Setup

```bash
source venv/bin/activate
```

The local `venv/` directory contains the Python 3.12 virtual environment.

## Running Code

- **Euler solver script:** `python euler1D.py` — runs all 6 test cases (Sod, Einfeldt, vacuum, strong shock, contact, Woodward-Colella) and saves comparison plots to `./outputs/`.
- **Notebooks:** Open `0-damp_osc.ipynb` or `1-ideal_riemann.ipynb` in Jupyter. Run cells sequentially.

There is no test suite, linter, or build system configured.

## Architecture

### `euler1D.py` — Classical Riemann Solvers

Self-contained module implementing three Godunov-type finite-volume Riemann solvers for the 1D Euler equations (ideal gas, gamma=1.4):

- **Exact solver** — scalar Newton iteration per interface, implemented in NumPy (`solve_star`, `_sample`, `exact_godunov_flux`)
- **Roe solver** — vectorized over all interfaces in JAX, includes Harten entropy fix (`roe_fluxes`)
- **HLL solver** — vectorized in JAX, uses Einfeldt/HLLE wave speed estimates (`hll_fluxes`)

The `solve()` function is the main time-stepper: first-order Godunov with CFL-limited timesteps, transmissive or reflecting boundary conditions. State is stored as conserved variables `[rho, rho*u, E]` in shape `[3, N]`.

Key conventions: `prim2cons`/`cons2prim` convert between primitive `(rho, u, p)` and conserved variables. Gas constants (`GM1`, `GP1`, `G1`–`G5`) are module-level derived from `GAMMA`. `FLOOR = 1e-12` prevents division by zero.

### `0-damp_osc.ipynb` — PINN for Damped Oscillator

Trains a physics-informed neural network (MLP with tanh activations) to solve the damped harmonic oscillator ODE. Loss = ODE residual + IC penalty. Uses JAX autodiff for computing derivatives of the network output w.r.t. time. Inputs: `(t, gamma, omega0)`.

### `1-ideal_riemann.ipynb` — NN for Riemann Star-State

Trains an MLP (SiLU activations) to predict `log10(p*)` from `(log10(rhoL), log10(pL), log10(rhoR), log10(pR), uRL)`. The loss is the squared residual of the pressure function `f(p*)` — no labeled data needed, only the physical constraint that `f(p*) = 0`. Includes a JAX-based Newton solver (`find_pstar`) for ground-truth comparison. Uses gamma=5/3.

## Key Conventions

- JAX 64-bit mode is enabled via `jax.config.update("jax_enable_x64", True)` in `euler1D.py`.
- Neural networks are implemented as raw param lists (no framework), trained with optax.adam.
- Log-space inputs are used for the Riemann NN to handle the wide dynamic range of gas states.
