# Handoff: continuing the SIREN attempt at 1e-9 p* prediction

You are picking up a task from a previous Claude session. Read this whole file before doing anything.

## The task

Train a neural network to predict `p*` (the Riemann star-region pressure) from the 5D gas state `(log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)` to **absolute error ≤ 1e-9 over the domain** defined in `experiments/adam_then_lbfgs_normmlp.py`:

```
log_rho_range = (0.0, 2.0)
log_p_range   = (0.0, 2.0)
u_range       = (-0.5, 0.5)
```

`p*` ranges over roughly `[0.1, 120]` on this domain, so 1e-9 absolute ≈ 1e-11 relative — tight but still well above float64 noise.

## Hard constraints (from the user)

1. **No Newton iterations wrapped around the MLP.** The model must be pure feed-forward at inference. `cascade` (MLP₁ · 10^(MLP₂ · …)) is allowed because it's feed-forward.
2. **No analytic guess as the starting point.** Specifically banned: `two_rarefaction_p0`, PVRS, or any physics-derived initial estimate fed into or anchoring the network.
3. **At most two models training simultaneously.** Empirically the CPU is bottlenecked — two parallel runs each go ~half speed. Serial is usually faster wall-clock. The session CPU is single device (`jax.devices()` → one `CpuDevice`).

## Current best

`outputs/cascade/stage1.msgpack` is a copy of the existing 100k-Adam-epoch `StarPressureMLPNormalized(width=256, depth=3)` checkpoint (originally trained with `fstar` residual loss on a wider `u_range=(-1,1)` domain). Evaluated on the target narrow domain against the precise solver:

```
abs_err_p50 = 2.90e-4
abs_err_p95 = 1.98e-3
abs_err_max = 1.97e-2
log_ratio_p95 = 4.84e-5
```

That's ~6 orders short of 1e-9.

## Important infrastructure already in place — use this, don't rebuild

- **`riemann_pinn/physics.py` — `find_pstar_precise`**: the existing `find_pstar` stops Newton at |f|<1e-6, capping ground-truth precision at ~1e-6 in p*. `find_pstar_precise` does 3 extra Newton polish steps, giving |f| ≤ 1e-15 (machine epsilon). **Always use `find_pstar_precise` for ground truth** — without it, any target below 1e-6 is noise.

- **`riemann_pinn/cascade.py`**: `CorrectionMLP`, `CascadeModel`, supervised log-MSE loss, LBFGS / SGD train step helpers, load/save utilities.

- **`riemann_pinn/model.py` — `SirenMLP`**: the SIREN architecture class. **It's broken in its current form** (see below for what to fix).

- **`cascade_stage.py`**: trains one residual cascade stage on top of a loaded frozen base. Reuse this framework once SIREN itself works.

- **`supervised_floor.py`**: trains a single MLP from scratch with supervised log-MSE loss on precise ground truth. Good template for SIREN training.

## What I already know doesn't work

1. **Plain MLP L-BFGS polish** of the 100k baseline with supervised log-MSE: plateaus immediately (~3% reduction over 100 steps). The baseline is already near-optimal in its function class.
2. **Residual cascade with MLP correction (w=256 d=3 or w=512 d=5)**: a fresh zero-init `CorrectionMLP` only captures ~5–15% of the residual variance. The residual of a trained MLP appears noise-like to another MLP of similar capacity.
3. **Cascade Adam LR=3e-4**: overshoots when the target is tiny (~1e-5). Zero-init final layer + Adam's sign-scaled updates push params past the target scale in one step.

So cascade-of-MLPs is probably not going to close the 6-order gap by itself.

## Why SIREN is the next bet

SIREN (Sitzmann et al., 2020) uses sinusoidal activations and a specific init. On smooth function fitting it reaches ~1e-4 to 1e-8 MSE where ReLU/SiLU MLPs plateau at 1e-2 to 1e-4. If it can hit log-MSE ~1e-15 here (abs_err ~3e-6), cascade might push the last 3 orders.

## My broken SIREN and how to fix it

`riemann_pinn/model.py` has a `SirenMLP` class. When I tried to train it (`width=256, depth=4, w0=10, lr=5e-4`), the loss stayed pinned at 1.385 — the network was not learning at all.

### The two bugs

1. **Output layer: `return 10.0 ** out`** — SIREN expects a plain linear output. The `10**x` non-linearity at the end breaks the SIREN analysis and the initialization assumptions. **Fix: return `out` directly (log10 p*), and have the loss compare against `log10(p_true)` without ever going back to physical space.**
2. **`w0=10` is too low.** The paper uses `w0=30` for image fitting. For this 5D smooth function you should start there. Lower `w0` → sines too slow → nothing distinguishes samples → loss stays at the variance of the target (which is ~1.4 for `log10(p*)` on this domain, matching what I saw).

### Other likely issues to watch

- **Learning rate**: SIREN is sensitive. Start at `1e-4`, not `5e-4`. If it doesn't move, try `5e-5`.
- **Batch size**: 4096 is fine.
- **Initialization**: already correct in the class (uniform with `±1/n_in` on layer 0, `±√(6/n_in)/w0` on hidden and output). Don't touch unless profiling says otherwise.

### Concrete fix to `SirenMLP`

In `riemann_pinn/model.py`, the class currently ends with:
```python
out = nn.Dense(self.output_dim, kernel_init=siren_init_hidden)(h)
if self.output_dim == 1:
    out = out.squeeze(-1)
return 10.0 ** out
```

Change to:
```python
out = nn.Dense(self.output_dim, kernel_init=siren_init_hidden)(h)
if self.output_dim == 1:
    out = out.squeeze(-1)
return out  # log10(p*), not p*
```

Then write a matching loss that consumes log p* directly — don't use `riemann_pinn.cascade.supervised_log_loss` as-is, because it does `jnp.log10(p_pred)` on the output. Write a variant:

```python
def supervised_log_loss_raw(params, apply_fn, gas_log):
    log_p_pred = apply_fn({"params": params}, gas_log)  # already log10 p*
    gs_p = physics.gas_log_to_phys(gas_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_p)
    diff = log_p_pred - jnp.log10(p_true)
    return jnp.mean(diff ** 2), {"loss/log_mse": jnp.mean(diff ** 2)}
```

## Recommended first move

1. Apply the two `SirenMLP` fixes above (remove `10**x`, and when you call it use `w0=30`).
2. Copy `supervised_floor.py` → `train_siren.py` with:
   - Use `SirenMLP(width=256, depth=4, w0=30)` and the `_raw` loss above.
   - 5000 Adam epochs at `lr=1e-4` (cosine decay to 1e-6), batch 4096.
   - Don't bother with L-BFGS yet — just verify it learns.
3. Run it. Expect loss to drop below ~1e-3 in the first few thousand steps. If it doesn't, the fix didn't take — re-verify init and w0.
4. Only once learning is working: scale to `depth=5`, more epochs, add L-BFGS polish with a fixed batch of 16384–65536, memory_size 50.

If SIREN-from-scratch reaches log-MSE ~1e-12 (abs_err ~3e-6), then stacking 1–2 cascade stages of SIREN correction (same architecture, zero-init-final) should get you into the 1e-9 range. The cascade plumbing in `cascade_stage.py` already supports arbitrary base forward functions — just pass in a lambda that calls the trained SIREN's `apply_fn`, but you may need to modify `_load_base` in that script to understand the SIREN base (it currently only handles `StarPressureMLPNormalized`).

## Running things

- Python: `/Users/abeane/Projects/riemaNN/venv/bin/python`
- Run from the worktree dir: `/Users/abeane/Projects/riemaNN/.claude/worktrees/jesus-take-the-wheel/`
- Background long jobs with `run_in_background: true`. **Do not pipe to `tail -1` or similar** — that buffers until EOF and you see nothing. Use `-u` for unbuffered python and let stdout flow to the task output file.
- Monitor progress with the `Monitor` tool using a `grep --line-buffered -E "..."` filter on the output file. Filter for `Adam done|lbfgs ep [0-9]+:|Metrics:|Traceback|Error:` etc. so you catch both progress and crashes.

## Throughput reference (CPU, JAX float64)

- `StarPressureMLPNormalized(w=256, d=3)` + batch=2048 supervised: ~78 it/s solo, ~15–28 it/s with two concurrent jobs
- `w=512, d=5` + batch=4096: ~5–6 it/s solo
- Running two jobs concurrently roughly halves each job's throughput on this machine. **Serial is usually faster wall-clock** despite the user's "two at a time" budget.

## Files that matter

Kept:
- `riemann_pinn/physics.py` — added `find_pstar_precise`
- `riemann_pinn/model.py` — contains broken `SirenMLP` (fix as above) and untested `FourierFeatureMLP`
- `riemann_pinn/cascade.py` — cascade module
- `cascade_stage.py`, `cascade_train.py` — cascade training scripts
- `supervised_floor.py` — single-MLP supervised training
- `outputs/cascade/stage1.msgpack` — copy of the 100k baseline (for cascade base)
- `experiments/adam_then_lbfgs_normmlp.py` — the target-domain experiment file

Pre-existing (don't modify without reason):
- `run.py`, `riemann_pinn/train.py`, `riemann_pinn/experiment.py`, `riemann_pinn/plot.py`
- `outputs/adam_then_lbfgs_normmlp_100k/` in the **main repo root** (not the worktree) has the original 100k checkpoint and plots. Useful to refer back to; don't write into it.

## One more thing

The user wants you to **keep going on SIREN**. Don't pivot to other architectures unless SIREN clearly caps out. If it does cap out before 1e-9, the next most promising ideas (in my guess order) are:

1. Cascade of SIREN modules
2. Fourier-features MLP (class stub exists in `model.py`, untested)
3. Multi-resolution / domain decomposition (split 5D domain into tiles, train per-tile, blend)

Good luck.
