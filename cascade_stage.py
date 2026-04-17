"""Train one additional residual-cascade stage on top of an arbitrary frozen
base model.  Clean, self-contained, uses supervised log-MSE loss on precise
ground truth, Adam + long LBFGS.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import jax
import optax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

from riemann_pinn import physics
from riemann_pinn.cascade import (
    CorrectionMLP,
    make_cascade_lbfgs_step,
    make_cascade_train_step,
    run_cascade_loop,
)
from riemann_pinn.model import StarPressureMLPNormalized
from riemann_pinn.train import create_train_state, load_checkpoint, uniform_log

DOMAIN = dict(log_rho_range=(0.0, 2.0), log_p_range=(0.0, 2.0), u_range=(-0.5, 0.5))


class CascadeStage(nn.Module):
    """p_out(x) = base_forward(x) * 10**eps(x).  eps is a trainable MLP with
    zero-init final layer so untrained output equals base_forward.
    """
    base_forward: Callable = None
    width: int = 256
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        p_base = jax.lax.stop_gradient(self.base_forward(x))
        eps = CorrectionMLP(width=self.width, depth=self.depth)(x)
        return p_base * (10.0 ** eps)


def supervised_log_loss(params, apply_fn, gas_log):
    p_pred = apply_fn({"params": params}, gas_log)
    gs_p = physics.gas_log_to_phys(gas_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_p)
    diff = jnp.log10(p_pred) - jnp.log10(p_true)
    loss = jnp.mean(diff ** 2)
    return loss, {"loss/log_mse": loss}


def evaluate(forward, seed=999, n=50_000):
    rng = jr.PRNGKey(seed)
    gs_log = uniform_log(rng, n, **DOMAIN)
    gs_phys = physics.gas_log_to_phys(gs_log)
    p_nn = forward(gs_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gs_phys)
    abs_err = jnp.abs(p_nn - p_true)
    log_ratio = jnp.abs(jnp.log10(p_nn / p_true))
    return {
        "abs_err_p50": float(jnp.percentile(abs_err, 50)),
        "abs_err_p95": float(jnp.percentile(abs_err, 95)),
        "abs_err_p99": float(jnp.percentile(abs_err, 99)),
        "abs_err_max": float(jnp.max(abs_err)),
        "log_ratio_p50": float(jnp.percentile(log_ratio, 50)),
        "log_ratio_p95": float(jnp.percentile(log_ratio, 95)),
        "log_ratio_max": float(jnp.max(log_ratio)),
    }


def _load_base(path: Path, base_kind: str, base_width: int, base_depth: int,
               base_stages: list[tuple[Path, int, int]]):
    """Build a frozen forward fn for the base chain.

    base_kind: kind of the stage-0 model. Currently only 'normmlp' supported.
    base_stages: list of (ckpt, width, depth) for extra CorrectionMLP stages.
    """
    if base_kind == "normmlp":
        model0 = StarPressureMLPNormalized(width=base_width, depth=base_depth)
    else:
        raise ValueError(f"unknown base_kind {base_kind}")

    # Try a few optimizer templates until one matches.
    candidates = [
        optax.chain(optax.clip_by_global_norm(1.0),
                    optax.adamw(learning_rate=optax.cosine_decay_schedule(4e-3, 100_000, 1e-6),
                                weight_decay=1e-4)),
        optax.lbfgs(learning_rate=1.0, memory_size=50),
    ]
    last_err = None
    for tx in candidates:
        try:
            tmpl = create_train_state(jr.PRNGKey(0), model0, tx, batch_size_hint=2048)
            s = load_checkpoint(path, tmpl)
            base_state = s
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"could not load base from {path}: {last_err}")

    def fwd0(x):
        return base_state.apply_fn({"params": base_state.params}, x)

    # Stack correction MLPs.
    fwd = fwd0
    for ckpt, w, d in base_stages:
        mlp = CorrectionMLP(width=w, depth=d)
        dummy = jnp.ones((2, physics.GAS_STATE_DIM))
        p0 = mlp.init(jr.PRNGKey(1), dummy)["params"]
        tmpl = TrainState.create(apply_fn=mlp.apply, params=p0,
                                 tx=optax.adam(1e-3))
        st = load_checkpoint(ckpt, tmpl)
        # Close mlp and st.params into the lambda
        fwd = (lambda prev, _mlp=mlp, _p=st.params:
               (lambda x: prev(x) * (10.0 ** _mlp.apply({"params": _p}, x))))(fwd)

    return fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base-path", required=True, type=Path)
    ap.add_argument("--base-kind", default="normmlp")
    ap.add_argument("--base-width", type=int, default=256)
    ap.add_argument("--base-depth", type=int, default=3)
    ap.add_argument("--base-stage", action="append", default=[],
                    help="Extra correction stage: path:width:depth")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--adam-epochs", type=int, default=10000)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-epochs", type=int, default=5000)
    ap.add_argument("--lbfgs-batch", type=int, default=65536)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_stages = []
    for s in args.base_stage:
        p, w, d = s.split(":")
        base_stages.append((Path(p), int(w), int(d)))

    print(f"Loading base: {args.base_path}")
    base_forward = _load_base(args.base_path, args.base_kind,
                              args.base_width, args.base_depth, base_stages)
    base_metrics = evaluate(base_forward)
    print("Base metrics:", json.dumps(base_metrics, indent=2))

    # Fresh trainable cascade stage.
    model = CascadeStage(base_forward=base_forward, width=args.width, depth=args.depth)

    rng = jr.PRNGKey(args.seed)

    if args.adam_epochs > 0:
        schedule = optax.cosine_decay_schedule(args.adam_lr, args.adam_epochs, 1e-6)
        tx = optax.chain(optax.clip_by_global_norm(1.0),
                         optax.adamw(learning_rate=schedule, weight_decay=0.0))
    else:
        tx = optax.identity()

    dummy = jnp.ones((args.batch_size, physics.GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def sampler(key, bs):
        return uniform_log(key, bs, **DOMAIN)

    # Initial metrics (should match base since eps=0 initially)
    def fwd0(x):
        return state.apply_fn({"params": state.params}, x)
    init_metrics = evaluate(fwd0)
    print("Initial (should match base):", json.dumps(init_metrics, indent=2))

    t0 = time.monotonic()
    if args.adam_epochs > 0:
        step = make_cascade_train_step(supervised_log_loss)
        state, losses = run_cascade_loop(
            state, step, sampler, jr.PRNGKey(args.seed + 1),
            n_epochs=args.adam_epochs, batch_size=args.batch_size,
            desc=f"{args.name} adam", log_every=200,
        )
        print(f"Adam done in {time.monotonic() - t0:.1f}s, final={float(losses[-1]):.3e}")

    # LBFGS polish on a fixed huge batch.
    if args.lbfgs_epochs > 0:
        lbfgs_tx = optax.lbfgs(learning_rate=1.0, memory_size=args.lbfgs_memory)
        state = TrainState.create(apply_fn=state.apply_fn, params=state.params, tx=lbfgs_tx)
        step = make_cascade_lbfgs_step(supervised_log_loss)
        fixed_batch = sampler(jr.PRNGKey(args.seed + 9999), args.lbfgs_batch)
        for ep in range(args.lbfgs_epochs):
            state, loss, _ = step(state, fixed_batch)
            if ep % 100 == 0 or ep == args.lbfgs_epochs - 1:
                print(f"  lbfgs ep {ep}: loss={float(loss):.3e}")

    dt = time.monotonic() - t0
    print(f"Total time: {dt:.1f}s")

    final_metrics = evaluate(lambda x: state.apply_fn({"params": state.params}, x))
    print("Final metrics:", json.dumps(final_metrics, indent=2))

    # Save just the CorrectionMLP params as a standalone state, so it can be
    # chained into future stages as --base-stage path:w:d.
    out_dir = Path("outputs/cascade")
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_params = state.params["CorrectionMLP_0"]
    corr_mlp = CorrectionMLP(width=args.width, depth=args.depth)
    dummy = jnp.ones((2, physics.GAS_STATE_DIM))
    corr_init = corr_mlp.init(jr.PRNGKey(0), dummy)["params"]
    bare_state = TrainState.create(
        apply_fn=corr_mlp.apply, params=corr_params,
        tx=optax.adam(1e-3),
    )
    (out_dir / f"{args.name}.msgpack").write_bytes(to_bytes(bare_state))

    meta = {**vars(args), "base_metrics": base_metrics, "init_metrics": init_metrics,
            "final_metrics": final_metrics, "training_time_s": round(dt, 1)}
    with (out_dir / f"{args.name}_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
