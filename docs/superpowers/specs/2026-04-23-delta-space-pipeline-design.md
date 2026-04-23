# Delta-space pipeline — design

**Date:** 2026-04-23
**Branch:** `dimens-reduc`
**Status:** design approved, awaiting spec review

## Goal

Collapse the training pipeline from its current 5-dimensional
`(log₁₀ρ_L, log₁₀p_L, log₁₀ρ_R, log₁₀p_R, u_RL)` representation to a
3-dimensional `(Δρ, Δp, Δu)` representation everywhere, and cut all the
redundant physics helpers that existed only to bridge the two
representations. The Riemann problem is scale-invariant under a
2-parameter symmetry group (rescale ρ, rescale p); the remaining 3
dimensionless parameters are the natural input and the only one the
network should ever see.

## Representation

Gas state is a length-3 vector `(drho, dp, du)`:

- `drho = (ρ_R − ρ_L) / (ρ_R + ρ_L) ∈ [−1, 1]`
- `dp   = (p_R − p_L) / (p_R + p_L) ∈ [−1, 1]`
- `du   = u_RL / ducrit(drho, dp) ∈ [−∞, 1]`

The non-dimensionalization `p_ref = 1, ρ_ref = 1` is implicit:
`p_L = 1 − dp, p_R = 1 + dp, ρ_L = 1 − drho, ρ_R = 1 + drho`.
Sound speeds, `ducrit`, and `p*` are all dimensionless (ratios to
`p_ref`). `p*` output by the model is `p*/p_ref`.

`GAS_STATE_DIM = 3`. Variables in code are `drho`, `dp`, `du` (matching
names already used inside `physics.py`).

## `riemann_pinn/physics.py`

Keep:

- `GAMMA, ALPHA, BETA, MU, GAS_STATE_DIM = 3`
- `get_ducrit(drho, dp)` — unchanged.
- `ftilde(p, drho, dp, LR)` — unchanged.
- `fstar(p, gas_state)` — unchanged; already 3D-native.
- `dfstar_dp = jax.grad(fstar, argnums=0)` — unchanged.
- `_newton(gas_state, p0)` — unchanged.
- `_bisect(gas_state)` — unchanged (it calls `two_rarefaction_p0`, which
  we rewrite).
- `find_pstar(gas_state)` — unchanged body.

Rewrite:

- `two_rarefaction_p0(gas_state)` — native 3D form. No detour through
  reconstructed 5-vectors.

```python
@jax.jit
def two_rarefaction_p0(gas_state):
    """3D two-rarefaction p* guess (Toro eq. 4.46), dimensionless."""
    drho, dp, du = gas_state
    cL = jnp.sqrt(GAMMA * (1 - dp) / (1 - drho))
    cR = jnp.sqrt(GAMMA * (1 + dp) / (1 + drho))
    ducrit = get_ducrit(drho, dp)
    num = jnp.maximum(cL + cR - MU * du * ducrit, 1e-30)
    den = cL / (1 - dp) ** ALPHA + cR / (1 + dp) ** ALPHA
    return (num / den) ** (1.0 / ALPHA)
```

Delete:

- `sound_speed`
- `ref_sound_speed`
- `find_ustar` — dead code; referenced a non-existent `fjump`.
- `gas_log_to_phys`
- `gas_phys_to_log`
- `two_rarefaction_p0_batch` — unused after the loss cuts.

## `riemann_pinn/model.py`

Replace both existing classes with one:

```python
class PressureMLP(nn.Module):
    """Maps (B, 3) delta state -> (B,) p*/p_ref, positive by log output."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):  # x is (B, 3) = (drho, dp, du)
        model = _MLP(width=self.width, depth=self.depth,
                     activation=self.activation, output_dim=1)
        return 10.0 ** model(x).squeeze(-1)
```

The log-space output (`10.0 ** …`) guarantees `p* > 0` by construction
and matches the ~1–2 order-of-magnitude dynamic range of `p*/p_ref`
across the sampling domain.

Delete the old 5D `PressureMLP` (all three normalize modes) and
`CompactDimensionPressureMLP`. Reusing the `PressureMLP` name is
deliberate: there is only one model class after the cut.

## `riemann_pinn/train.py`

**Samplers** — 3D, keys `drho_range, dp_range, du_range`:

```python
def uniform(rng, batch_size, *, drho_range, dp_range, du_range) -> jnp.ndarray: ...
def r2_quasirandom(rng, batch_size, *, drho_range, dp_range, du_range) -> jnp.ndarray: ...
```

Both return shape `(B, 3)`. R2 uses `_R2_GOLDEN[2]` with `NDIM = 3`.
`uniform` replaces the old `uniform_log` name. Per-call keyword
arguments mirror the domain dict keys exactly.

**Losses** — kept to three:

- `residual_loss(params, apply_fn, gas_states)` — `mean(fstar(p*_NN, state)²)`.
  Direct; no log→phys conversion layer.
- `residual_loss_newton(params, apply_fn, gas_states)` — `mean((f/f')²)`.
- `supervised_loss(params, apply_fn, gas_states)` — `mean((p*_NN − p*_true)²)`
  with `p*_true` from `find_pstar(gas_state)`.

Delete `residual_loss_normalized` — its `ref_sound_speed`
normalization is redundant once `fstar` is dimensionless.

**Domain schema** — `Experiment.domain` and `Experiment.train_domain`
now use keys `drho_range, dp_range, du_range`. Default domain used by
`evaluate_holdout` (and by the smoke test):

```python
dict(drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9))
```

**`Experiment` / `Phase` dataclasses** — structurally unchanged.
`create_train_state` uses `GAS_STATE_DIM = 3` for its dummy input.

**`evaluate_holdout`** — sampler call switches to `uniform(...)` with
the new kwargs; `gas_log_to_phys` calls removed; `fstar`, `find_pstar`,
etc. take the sampled 3D state directly.

## `riemann_pinn/plot.py`

All plots operate directly on 3D gas states. No `gas_log_to_phys` calls
anywhere.

- `plot_loss` — unchanged.
- `plot_slice(state, out_path, *, n, drho_range, dp_range, du_slice=0.0, err_range, nbins, name)`
  — three panels:
  1. `log₁₀(p*_NN / p*_true)` heatmap over `(drho, dp)` at `du = du_slice`.
  2. `sign(f)·log₁₀|f(p*_NN)|` heatmap over the same grid.
  3. Histogram of `log₁₀(p*_NN / p*_true)`, with an overlaid second
     histogram comparing `p*(drho, dp, du)` against
     `p*(−drho, −dp, +du)` — the physical L↔R symmetry of the 1D
     Riemann problem (spatial reflection: swap L/R labels **and**
     negate velocities). Because `uRL = uR − uL` is invariant under
     this reflection (both velocities flip sign and the labels
     swap), `du = uRL / ducrit` is also invariant; only `drho` and
     `dp` flip. The true `p*` is unchanged under this swap, so the
     overlay histogram should peak sharply at zero for a well-trained
     network. Note: the old 5D pipeline's slice plot used an
     incorrect swap that negated `uRL` as well — this was a bug
     that did not correspond to any physical invariance.
- `plot_pstar_hist2d` — unchanged idea (true vs NN), 3D sampling.
- `_corner_panels` — rewritten for 3 vars, returns 3 panels:
  `(drho, dp), (drho, du), (dp, du)`.
- `plot_corner_error`, `plot_corner_pstar` — 2×2 grid with 3 populated
  tiles. `_VAR_LABELS = [r"$\Delta\rho$", r"$\Delta p$", r"$\Delta u$"]`.

## `run.py`

Unchanged except for the `plot_slice` call site — passes `drho_range`
and `dp_range` from `exp.domain` instead of the old `log_rho_range` /
`log_p_range`.

## Experiments and top-level scripts

Move every file currently directly under `experiments/` into
`experiments/archive/`. This includes:
`adam_then_lbfgs_normloss.py`, `adam_then_lbfgs_normmlp_100k.py`,
`adam_then_lbfgs_normmlp_newton.py`, `adam_then_lbfgs_normmlp.py`,
`adam_then_lbfgs.py`, `adamw_dreducmlp.py`, `adamw_normmlp_geommean.py`,
`adamw_normmlp_geom_widen_L6.py`, `adamw_normmlp_geom_widen.py`,
`adamw_normmlp.py`, `adamw_normmlp_wdgrid.py`, `adamw.py`,
`lbfgs_ladder.py`, `lbfgs_small.py`, `smoke_test.py`.

These stay in `archive/` as historical record. They use deleted
imports and are not expected to run after the refactor.

Delete `experiments_hipres/` directory entirely (only one file, already
5D).

Write one fresh experiment file:

- `experiments/smoke_test.py` — one `Experiment` named
  `"smoke_test"`. Model `PressureMLP(width=64, depth=2)`. Domain
  `drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9)`.
  One `Phase`: AdamW with small cosine schedule, ~1000 epochs,
  `loss=supervised_loss`, `sampler=uniform`. Minimal working example
  and default run target.

Delete `check_vacuum.py` and `test_prime.py` — both are one-off
exploratory scripts that use deleted 5D APIs.

## `CLAUDE.md`

Rewrite Architecture and Key Conventions sections to match:

- Remove stale `fjump` reference.
- `GAS_STATE_DIM = 5` → `3`.
- Domain description: log-space ranges → `drho_range, dp_range, du_range` with the new default values.
- Input/sampling description: the network input is `(drho, dp, du)`, not a log-space 5-vector.
- Loss list: `residual_loss`, `residual_loss_newton`, `supervised_loss`
  (drop `residual_loss_normalized`).
- Sampler list: `uniform`, `r2_quasirandom` (drop `uniform_log`).

## Non-goals

- No changes to checkpoint format (it is a serialized Flax
  `TrainState`; format is driven by the model class shape, which
  changes, so existing `outputs/*/checkpoint.msgpack` files are
  invalidated — this is expected under the hard break).
- No changes to `report.py` (it reads `metrics.json` files and is
  independent of representation).
- No changes to notebooks under `note/` (standalone, no shared code).
- No attempt at backward-compatibility shims: every 5D import is
  removed, not aliased.

## Verification

After the refactor, the following must hold:

1. `venv/bin/python -c "from riemann_pinn import physics, model, train, plot"`
   imports with no errors.
2. `grep -r "gas_log_to_phys\|gas_phys_to_log\|sound_speed\|ref_sound_speed\|find_ustar\|CompactDimensionPressureMLP\|residual_loss_normalized\|uniform_log\|log_rho_range\|log_p_range\|u_range" riemann_pinn/ run.py experiments/smoke_test.py`
   returns no matches (the 5D API is fully excised from the live code
   paths; `experiments/archive/` is allowed to still contain these).
3. `venv/bin/python run.py experiments/smoke_test.py --retrain` trains
   for ~1000 epochs without NaNs and produces `checkpoint.msgpack`,
   `loss.npy`, `metrics.json`, and all five plot files under
   `outputs/smoke_test/smoke_test/`.
4. Holdout metrics on the smoke test show `p95_abs_fstar < 1e-1` after
   1000 Adam steps (a sanity floor, not a quality target).
