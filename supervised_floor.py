"""Train a fresh MLP with direct supervised log-MSE loss on the precise
ground-truth p*. Measures the single-MLP precision floor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import optax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.training.train_state import TrainState

from riemann_pinn import physics
from riemann_pinn.cascade import (
    make_cascade_lbfgs_step,
    make_cascade_train_step,
    run_cascade_loop,
)
from riemann_pinn.model import StarPressureMLPNormalized, StarPressureMLP
from riemann_pinn.train import create_train_state, uniform_log

DOMAIN = dict(log_rho_range=(0.0, 2.0), log_p_range=(0.0, 2.0), u_range=(-0.5, 0.5))


def supervised_log_loss(params, apply_fn, gas_log):
    p_pred = apply_fn({"params": params}, gas_log)
    gs_p = physics.gas_log_to_phys(gas_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_p)
    diff = jnp.log10(p_pred) - jnp.log10(p_true)
    loss = jnp.mean(diff ** 2)
    return loss, {"loss/log_mse": loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="supervised_floor")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--adam-epochs", type=int, default=30_000)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--lbfgs-epochs", type=int, default=1000)
    ap.add_argument("--lbfgs-batch", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm", action="store_true", help="use StarPressureMLPNormalized instead of plain")
    args = ap.parse_args()

    if args.norm:
        model = StarPressureMLPNormalized(width=args.width, depth=args.depth)
    else:
        model = StarPressureMLP(width=args.width, depth=args.depth)

    # Adam phase.
    schedule = optax.cosine_decay_schedule(args.adam_lr, args.adam_epochs, 1e-5)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(learning_rate=schedule, weight_decay=0.0))
    state = create_train_state(jr.PRNGKey(args.seed), model, tx, batch_size_hint=args.batch_size)

    def sampler(key, bs):
        return uniform_log(key, bs, **DOMAIN)

    step = make_cascade_train_step(supervised_log_loss)
    t0 = time.monotonic()
    state, losses = run_cascade_loop(
        state, step, sampler, jr.PRNGKey(args.seed + 1),
        n_epochs=args.adam_epochs, batch_size=args.batch_size,
        desc=f"{args.name} adam", log_every=500,
    )
    adam_dt = time.monotonic() - t0
    print(f"Adam done in {adam_dt:.1f}s, final loss={float(losses[-1]):.3e}")

    # LBFGS polish
    if args.lbfgs_epochs > 0:
        lbfgs_tx = optax.lbfgs(learning_rate=1.0, memory_size=20)
        state = TrainState.create(apply_fn=state.apply_fn, params=state.params, tx=lbfgs_tx)
        step = make_cascade_lbfgs_step(supervised_log_loss)
        fixed_batch = sampler(jr.PRNGKey(args.seed + 9999), args.lbfgs_batch)
        losses_lbfgs = []
        for ep in range(args.lbfgs_epochs):
            state, loss, _ = step(state, fixed_batch)
            losses_lbfgs.append(float(loss))
            if ep % 50 == 0 or ep == args.lbfgs_epochs - 1:
                print(f"  lbfgs ep {ep}: loss={float(loss):.3e}")

    dt = time.monotonic() - t0

    # Evaluate
    rng = jr.PRNGKey(999)
    gs_log = uniform_log(rng, 50_000, **DOMAIN)
    gs_phys = physics.gas_log_to_phys(gs_log)
    p_nn = state.apply_fn({"params": state.params}, gs_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_phys)
    abs_err = jnp.abs(p_nn - p_true)
    log_ratio = jnp.abs(jnp.log10(p_nn / p_true))
    metrics = {
        "abs_err_p50": float(jnp.percentile(abs_err, 50)),
        "abs_err_p95": float(jnp.percentile(abs_err, 95)),
        "abs_err_p99": float(jnp.percentile(abs_err, 99)),
        "abs_err_max": float(jnp.max(abs_err)),
        "log_ratio_p50": float(jnp.percentile(log_ratio, 50)),
        "log_ratio_p95": float(jnp.percentile(log_ratio, 95)),
        "log_ratio_max": float(jnp.max(log_ratio)),
        "training_time_s": round(dt, 1),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # Save
    out_dir = Path("outputs/cascade")
    out_dir.mkdir(parents=True, exist_ok=True)
    from flax.serialization import to_bytes
    (out_dir / f"{args.name}.msgpack").write_bytes(to_bytes(state))
    np.save(out_dir / f"{args.name}_loss.npy", np.asarray(losses))
    with (out_dir / f"{args.name}_meta.json").open("w") as f:
        json.dump({**vars(args), **metrics}, f, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
