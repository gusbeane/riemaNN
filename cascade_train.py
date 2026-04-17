"""Train one residual-cascade stage on top of an existing chain.

Usage:
  venv/bin/python cascade_train.py stage2 \
      --stage1 outputs/cascade/stage1.msgpack \
      --adam-epochs 30000 --adam-lr 4e-3 --lbfgs-epochs 500 \
      --width 256 --depth 3 --batch-size 4096

The first positional arg is the stage name (e.g. stage2). The chain built so
far is given as --stage1, --stage2, etc. This script trains one NEW correction
MLP that sits on top of the given chain and writes its params to
outputs/cascade/<name>.msgpack, plus loss.npy and metrics.json.

Domain is hardcoded to match experiments/adam_then_lbfgs_normmlp.py:
  log_rho in [0,2], log_p in [0,2], u in [-0.5, 0.5].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import optax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from riemann_pinn import physics  # noqa: E402
from riemann_pinn.cascade import (  # noqa: E402
    CascadeModel,
    CorrectionMLP,
    cascade_supervised_log_loss,
    cascade_supervised_abs_loss,
    create_cascade_train_state,
    make_cascade_lbfgs_step,
    make_cascade_train_step,
    run_cascade_loop,
    save_stage,
    load_stage,
    compose_chain,
)
from riemann_pinn.model import StarPressureMLPNormalized  # noqa: E402
from riemann_pinn.train import (  # noqa: E402
    create_train_state,
    uniform_log,
)
from flax.training.train_state import TrainState  # noqa: E402


DOMAIN = dict(log_rho_range=(0.0, 2.0), log_p_range=(0.0, 2.0), u_range=(-0.5, 0.5))


def _load_stage1(path: Path, width=256, depth=3):
    """Load the plain Stage-1 pstar model (trained with adamw + cosine).

    Uses a matching optimizer template so flax deserialization succeeds.
    The opt_state is not used afterwards.
    """
    model = StarPressureMLPNormalized(width=width, depth=depth)
    schedule = optax.cosine_decay_schedule(4e-3, 100_000, 1e-6)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )
    template = create_train_state(jr.PRNGKey(0), model, tx, batch_size_hint=2048)
    state = load_stage(path, template)
    def forward(x):
        return state.apply_fn({"params": state.params}, x)
    return forward


def _load_correction(path: Path, width: int, depth: int):
    """Load a CorrectionMLP stage checkpoint; return (model, params)."""
    mlp = CorrectionMLP(width=width, depth=depth)
    dummy = jnp.ones((2, physics.GAS_STATE_DIM))
    params_init = mlp.init(jr.PRNGKey(0), dummy)["params"]
    # Build a minimal train_state-compatible object for from_bytes
    tx = optax.adam(1e-3)
    tmpl = TrainState.create(apply_fn=mlp.apply, params=params_init, tx=tx)
    state = load_stage(path, tmpl)
    return mlp, state.params


def build_base_forward(stage1_path: Path,
                       extra_stage_paths: list[Path],
                       extra_widths: list[int],
                       extra_depths: list[int]):
    """Build the cumulative base forward from all previously-trained stages."""
    fwd = _load_stage1(stage1_path)
    for path, w, d in zip(extra_stage_paths, extra_widths, extra_depths):
        mlp, p = _load_correction(path, w, d)
        fwd = (lambda prev_fwd, mlp=mlp, p=p:
               (lambda x: prev_fwd(x) * (10.0 ** mlp.apply({"params": p}, x))))(fwd)
    return fwd


def evaluate(forward, n_samples=50_000, seed=999):
    rng = jr.PRNGKey(seed)
    gs_log = uniform_log(rng, n_samples, **DOMAIN)
    gs_phys = physics.gas_log_to_phys(gs_log)
    p_nn = forward(gs_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_phys)

    abs_err = jnp.abs(p_nn - p_true)
    log_ratio = jnp.abs(jnp.log10(p_nn) - jnp.log10(p_true))

    return {
        "abs_err_median": float(jnp.median(abs_err)),
        "abs_err_p50": float(jnp.percentile(abs_err, 50)),
        "abs_err_p95": float(jnp.percentile(abs_err, 95)),
        "abs_err_p99": float(jnp.percentile(abs_err, 99)),
        "abs_err_max": float(jnp.max(abs_err)),
        "log_ratio_median": float(jnp.median(log_ratio)),
        "log_ratio_p95": float(jnp.percentile(log_ratio, 95)),
        "log_ratio_max": float(jnp.max(log_ratio)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Output stage name, e.g. stage2 / stage3")
    ap.add_argument("--stage1", required=True, type=Path)
    ap.add_argument("--stage", action="append", type=str, default=[],
                    help="Extra stage checkpoints to include in base, e.g. 'path:w:d'")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--adam-epochs", type=int, default=30_000)
    ap.add_argument("--adam-lr", type=float, default=4e-3)
    ap.add_argument("--adam-alpha", type=float, default=1e-4)
    ap.add_argument("--lbfgs-epochs", type=int, default=500)
    ap.add_argument("--lbfgs-batch", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--loss", choices=["log", "abs"], default="log")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cascade"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    extra_paths, extra_ws, extra_ds = [], [], []
    for spec in args.stage:
        path, w, d = spec.split(":")
        extra_paths.append(Path(path))
        extra_ws.append(int(w))
        extra_ds.append(int(d))

    print(f"Building base forward from stage1={args.stage1} + {len(extra_paths)} correction stages")
    base_forward = build_base_forward(args.stage1, extra_paths, extra_ws, extra_ds)

    base_metrics = evaluate(base_forward)
    print("Base forward metrics:", json.dumps(base_metrics, indent=2))

    # Build trainable cascade model
    model = CascadeModel(base_forward=base_forward, width=args.width, depth=args.depth)

    loss_fn = cascade_supervised_log_loss if args.loss == "log" else cascade_supervised_abs_loss

    # Build initial train_state. If adam_epochs=0 we skip building the
    # cosine schedule (which requires decay_steps>0) and use a no-op optimizer.
    rng = jr.PRNGKey(args.seed)
    if args.adam_epochs > 0:
        schedule = optax.cosine_decay_schedule(
            init_value=args.adam_lr, decay_steps=args.adam_epochs, alpha=args.adam_alpha,
        )
        init_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=1e-6),
        )
    else:
        init_tx = optax.identity()
    state = create_cascade_train_state(rng, model, init_tx, batch_size_hint=args.batch_size)

    def sampler(key, batch):
        return uniform_log(key, batch, **DOMAIN)

    t0 = time.monotonic()
    if args.adam_epochs > 0:
        step = make_cascade_train_step(loss_fn)
        state, loss_trace_adam = run_cascade_loop(
            state, step, sampler, jr.fold_in(rng, 1),
            n_epochs=args.adam_epochs, batch_size=args.batch_size,
            desc=f"{args.name} adam", log_every=200,
        )
    else:
        loss_trace_adam = jnp.array([])

    # Eval after Adam
    def current_forward(x):
        return state.apply_fn({"params": state.params}, x)
    mid_metrics = evaluate(current_forward)
    print("After Adam:", json.dumps(mid_metrics, indent=2))

    # L-BFGS phase (fresh optimizer, same params)
    if args.lbfgs_epochs > 0:
        lbfgs_tx = optax.lbfgs(learning_rate=1.0, memory_size=20)
        state = TrainState.create(apply_fn=state.apply_fn, params=state.params, tx=lbfgs_tx)
        step = make_cascade_lbfgs_step(loss_fn)

        # Fixed big batch for LBFGS polish.
        fixed_batch = sampler(jr.PRNGKey(args.seed + 777), args.lbfgs_batch)
        losses_lbfgs = []
        pbar_rng = jr.PRNGKey(args.seed + 888)
        for ep in range(args.lbfgs_epochs):
            state, loss, _ = step(state, fixed_batch)
            losses_lbfgs.append(float(loss))
            if ep % 50 == 0:
                print(f"  lbfgs ep {ep}: loss={loss:.3e}")
        loss_trace_lbfgs = jnp.array(losses_lbfgs)
    else:
        loss_trace_lbfgs = jnp.array([])

    dt = time.monotonic() - t0
    print(f"Training done in {dt:.1f}s")

    # Final eval
    def final_forward(x):
        return state.apply_fn({"params": state.params}, x)
    final_metrics = evaluate(final_forward)
    print("Final metrics:", json.dumps(final_metrics, indent=2))

    # Save ONLY the correction MLP params (not the base_forward closure).
    # state.params has the form {"CorrectionMLP_0": {...}}
    ckpt_dir = args.out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.name}.msgpack"

    # Save just the correction subtree as a standalone state.
    corr_params = state.params["CorrectionMLP_0"]
    corr_mlp = CorrectionMLP(width=args.width, depth=args.depth)
    bare_state = TrainState.create(
        apply_fn=corr_mlp.apply, params=corr_params,
        tx=optax.adam(1e-3),
    )
    save_stage(ckpt_path, bare_state)
    print(f"Saved {ckpt_path}")

    # Save loss trace + metrics
    loss_trace = jnp.concatenate([loss_trace_adam, loss_trace_lbfgs])
    np.save(ckpt_dir / f"{args.name}_loss.npy", np.asarray(loss_trace))

    meta = {
        "name": args.name,
        "width": args.width,
        "depth": args.depth,
        "adam_epochs": args.adam_epochs,
        "adam_lr": args.adam_lr,
        "lbfgs_epochs": args.lbfgs_epochs,
        "lbfgs_batch": args.lbfgs_batch,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "training_time_s": round(dt, 1),
        "base_stage1": str(args.stage1),
        "base_extra": args.stage,
        "base_metrics": base_metrics,
        "mid_metrics": mid_metrics,
        "final_metrics": final_metrics,
    }
    with (ckpt_dir / f"{args.name}_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Saved {ckpt_dir / f'{args.name}_meta.json'}")


if __name__ == "__main__":
    main()
