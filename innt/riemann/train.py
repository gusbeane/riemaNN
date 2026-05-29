"""Train the integrated face-flux network F(t, uL, uR) = t * F_net(t, uL, uR).

Adam pretrain -> L-BFGS refine. Every `flush_every` steps (default 100) and at
the last step of each stage, append recent losses, evaluate the network on a
set of regimes, append a metrics row, and save a versioned msgpack checkpoint.

Burgers-specific code lives here. `evaluate.py` is physics-agnostic and
`checkpoint.py` is model-agnostic; both are driven by callbacks passed from
this module.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import argparse
import sys
from pathlib import Path
from typing import Mapping

import flax.nnx as nnx
import jax.numpy as jnp
import optax
from jax import random as jr
from tqdm import tqdm

# Make sibling modules importable when invoked as `python innt/burger/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import CheckpointWriter, load_latest  # noqa: E402
from evaluate import (  # noqa: E402
    draw_rect,
    draw_small_jumps,
    evaluate_all,
    metric_keys_for,
)
from physics import find_pstar, compute_flux, compute_integrated_flux, GAS_STATE_DIM, GasState

FLUX_SCALE = jnp.array([1e-5, 1e-3, 1e-5])

def F_pred(F_net, x):
    """Network's prediction of F(t, drho, dp, du) = t * F_net(t, drho, dp, du).

    Works for both single `x` (shape (4,)) and batched `x` (shape (N, 4)).
    Output is a 3-vector flux (mass, momentum, energy) per sample.
    """
    return x[..., 0:1] * F_net(x)


def F_true(x: jax.Array) -> jax.Array:
    """Exact integrated face flux at a single eval point.

    The star state at xi = 0 is t-invariant for the self-similar Riemann
    problem, so F(t) = integral_0^t f(u*(0)) ds = t * f(u*(0)). That product
    is exactly what `compute_flux` returns.
    """
    t = x[0]
    gs = GasState.from_array(x[1:])
    return compute_integrated_flux(t, gs)


# ---------------------------------------------------------------------------
# Model

class MLP(nnx.Module):
    def __init__(self, dims, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, x, activation=nnx.tanh):
        x = jnp.atleast_2d(x)
        n = len(self.layers)
        for i in range(n-1):
            x = activation(self.layers[i](x))
        
        return self.layers[-1](x).squeeze()


def init_nn(in_dim: int, width: int = 32, depth: int = 3, out_dim: int = 1, *, seed: int = 0) -> MLP:
    dims = [in_dim] + [width] * depth + [out_dim]
    return MLP(dims, rngs=nnx.Rngs(seed))


def _build_model(arch: dict) -> MLP:
    """Skeleton builder for `checkpoint.load_model`. Initial weights are overwritten."""
    return init_nn(**arch, seed=0)


# ---------------------------------------------------------------------------
# Sampler + training/eval bounds

# (t, drho, dp, du)
# TRAIN_BOUNDS      = jnp.array([[0., 1.0], [-0.8, 0.8], [-0.8, 0.8], [-1.0, 0.4]])
# FULL_BOUNDS       = jnp.array([[0., 1.0], [-0.8, 0.8], [-0.8, 0.8], [-1.0, 0.4]])
# RESTRICTED_BOUNDS = jnp.array([[0., 0.8], [-0.6, 0.6], [-0.6, 0.6], [-0.8, 0.3]])

# (t, log10_rhoL, log10_pL, log10_rhoR, log10_pR, uRL)
TRAIN_BOUNDS      = jnp.array([[0., 1.0], [-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.], [-1., 1.]])
FULL_BOUNDS       = jnp.array([[0., 1.0], [-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.], [-1., 1.]])
RESTRICTED_BOUNDS = jnp.array([[0., 0.8], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-0.8, 0.8]])


class UniformRandomSampler:
    """Uniform random sampler over a bounded rectangular domain."""

    def draw_batch(self, rng, batch_size: int, dim: int, bounds: jax.Array) -> jax.Array:
        lo, hi = bounds[:, 0], bounds[:, 1]
        u = jr.uniform(rng, (batch_size, dim), minval=0.0, maxval=1.0)
        return lo + (hi - lo) * u


REGIME_LABELS = ("full", "restricted", "small_jump")
METRIC_KEYS = metric_keys_for(REGIME_LABELS)



def make_eval_regimes(key: jax.Array, batch_size: int) -> dict[str, jax.Array]:
    k_full, k_r, k_sj = jr.split(key, 3)
    return {
        "full":       draw_rect(k_full, batch_size, FULL_BOUNDS),
        "restricted": draw_rect(k_r,    batch_size, RESTRICTED_BOUNDS),
        "small_jump": draw_small_jumps(k_sj, batch_size),
    }


def print_metrics(m: Mapping[str, float]) -> None:
    """Riemann-flavored, human-friendly metric summary.

    For each regime, prints the per-channel (mass/momentum/energy) median and
    max of the relative error |F_pred - F_true| / |F_true|.
    """
    channels = (
        ("mass", "mass_flux"),
        ("momentum", "momentum_flux"),
        ("energy", "energy_flux"),
    )

    for regime in REGIME_LABELS:
        tqdm.write(f"  {regime}:")
        for display_name, key_name in channels:
            tqdm.write(
                f"    {display_name:8s}: "
                f"median |rel|={m[f'median_rel_{regime}_{key_name}']:.2e}  "
                f"max |rel|={m[f'max_rel_{regime}_{key_name}']:.2e}"
            )
        tqdm.write("-" * 60)


# ---------------------------------------------------------------------------
# Loss

def loss(x: jax.Array, net, scale: float = 1.0) -> jax.Array:
    F = lambda r: r[..., 0] * net(r)
    F_t_eval = jax.vmap(jax.jacfwd(F))(x)[..., 0]

    compute_flux_of_x = lambda x: compute_flux(x[0], GasState.from_array(x[1:]))
    flux_eval = jax.vmap(compute_flux_of_x)(x)
    
    # return jnp.mean(jnp.log(jnp.abs(F_t_eval/flux_eval)))
    # return jnp.mean((jnp.log(F_t_eval) - jnp.log(flux_eval))**2)
    
    # return jnp.mean((jnp.arcsinh(F_t_eval / FLUX_SCALE) - jnp.arcsinh(flux_eval / FLUX_SCALE)) ** 2)
    
    return jnp.mean((F_t_eval - flux_eval) ** 2 / FLUX_SCALE**2)
    
    # err = (F_t_eval - flux_eval) / FLUX_SCALE
    # return jnp.mean(optax.huber_loss(err, delta=1.0))


# ---------------------------------------------------------------------------
# Training stages

@nnx.jit
def _adam_step(F_net: MLP, opt: nnx.Optimizer, batch: jax.Array) -> jax.Array:
    loss_val, grads = nnx.value_and_grad(lambda net: loss(batch, net))(F_net)
    opt.update(F_net, grads)
    return loss_val


@nnx.jit
def _lbfgs_step(F_net: MLP, opt: nnx.Optimizer, x_batch: jax.Array) -> jax.Array:
    def loss_fn(m):
        return loss(x_batch, m)

    graphdef, _params, rest = nnx.split(F_net, nnx.Param, ...)
    loss_val, grads = nnx.value_and_grad(loss_fn)(F_net)

    def value_fn(trial_params):
        return loss_fn(nnx.merge(graphdef, trial_params, rest))

    opt.update(F_net, grads, value=loss_val, grad=grads, value_fn=value_fn)
    return loss_val


def train_adam(
    F_net: MLP,
    *,
    writer: CheckpointWriter,
    n_steps: int,
    step_offset: int = 0,
    flush_every: int = 100,
    lr: float = 1e-3,
    batch_size: int = 100_000,
    seed: int = 0,
) -> None:
    print('lr:', lr)
    opt = nnx.Optimizer(F_net, optax.adamw(lr), wrt=nnx.Param)
    sampler = UniformRandomSampler()
    key = jr.PRNGKey(seed)

    pbar = tqdm(range(1, n_steps + 1), desc="adam")
    for i in pbar:
        subkey, key = jr.split(key)
        batch = sampler.draw_batch(subkey, batch_size, GAS_STATE_DIM+1, TRAIN_BOUNDS)
        loss_val = _adam_step(F_net, opt, batch)
        writer.record_loss(loss_val)

        global_step = step_offset + i
        if i % 10 == 0:
            pbar.set_description(f"adam loss={float(loss_val):.4e}")
        if global_step % flush_every == 0 or i == n_steps:
            writer.flush(global_step, "adam", F_net)
        if i==0:
            pbar.reset(total=n_steps-1)

def train_lbfgs(
    F_net: MLP,
    *,
    writer: CheckpointWriter,
    n_steps: int,
    step_offset: int = 0,
    flush_every: int = 100,
    batch_size: int = 2**17,
    batch_seed: int = 42,
) -> None:
    sampler = UniformRandomSampler()
    x_batch = sampler.draw_batch(jr.PRNGKey(batch_seed), batch_size, GAS_STATE_DIM+1, TRAIN_BOUNDS)
    opt = nnx.Optimizer(F_net, optax.lbfgs(), wrt=nnx.Param)

    pbar = tqdm(range(1, n_steps + 1), desc="lbfgs")
    for i in pbar:
        loss_val = _lbfgs_step(F_net, opt, x_batch)
        writer.record_loss(loss_val)

        global_step = step_offset + i
        if i % 10 == 0:
            pbar.set_description(f"lbfgs loss={float(loss_val):.4e}")
        if global_step % flush_every == 0 or i == n_steps:
            writer.flush(global_step, "lbfgs", F_net)
        if i==0:
            pbar.reset(total=n_steps-1)


# ---------------------------------------------------------------------------
# CLI

def _default_ckpt_dir() -> Path:
    return Path(__file__).resolve().parent / "checkpoints"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, default=_default_ckpt_dir())
    p.add_argument("--adam-steps", type=int, default=2000)
    p.add_argument("--lbfgs-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=2**17)
    p.add_argument("--adam-batch-size", type=int, default=2**17)
    p.add_argument("--lbfgs-batch-size", type=int, default=2**17)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--skip-lbfgs", action="store_true")
    args = p.parse_args(argv)

    arch = {"in_dim": GAS_STATE_DIM+1, "width": 32, "depth": 3, "out_dim": 3}
    F_net = init_nn(**arch, seed=args.seed)

    eval_key = jr.PRNGKey(args.eval_seed)

    def evaluate_fn(net: nnx.Module) -> dict[str, float]:
        regimes = make_eval_regimes(eval_key, args.eval_batch_size)
        return evaluate_all(
            net, F_pred_fn=F_pred, F_true_fn=F_true, regimes=regimes
        )

    writer = CheckpointWriter(
        args.ckpt_dir,
        arch,
        evaluate_fn=evaluate_fn,
        metric_keys=METRIC_KEYS,
        print_metrics_fn=print_metrics,
    )

    train_adam(
        F_net,
        writer=writer,
        n_steps=args.adam_steps,
        batch_size=args.adam_batch_size,
        step_offset=0,
        flush_every=args.flush_every,
        lr=args.adam_lr,
    )
    if not args.skip_lbfgs:
        train_lbfgs(
            F_net,
            writer=writer,
            n_steps=args.lbfgs_steps,
            batch_size=args.lbfgs_batch_size,
            step_offset=args.adam_steps,
            flush_every=args.flush_every,
        )

    reloaded = load_latest(args.ckpt_dir, _build_model)
    x = jr.uniform(jr.PRNGKey(7), (8, arch["in_dim"]), minval=-0.5, maxval=0.5)
    if not jnp.allclose(F_net(x), reloaded(x)):
        raise AssertionError("checkpoint round-trip mismatch")
    print("checkpoint round-trip OK")


if __name__ == "__main__":
    main()
