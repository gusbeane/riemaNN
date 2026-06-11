"""Train the integrated face-flux network F(t, uL, uR) = t * F_net(t, uL, uR).

Adam pretrain -> L-BFGS refine. Every `flush_every` steps (default 100) and at
the last step of each stage, append recent losses, evaluate the network on a
set of regimes, append a metrics row, and save a versioned msgpack checkpoint.

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
from physics import compute_flux, GAS_STATE_DIM, GasState, flux_from_primitive_state, GAMMA, abs_flux_jacobian_from_primitive_state

FLUX_SCALE_ARCSINH = jnp.array([1e-2, 1e-3, 2e-3])

from physics import rhoL_idx, uL_idx, pL_idx, rhoR_idx, uR_idx, pR_idx

def flux_scale(x: jax.Array) -> jax.Array:
    """Per-sample, per-channel characteristic Euler flux scale (mass, momentum, energy).

    Built from the average state: mass ~ rho_c a_c, momentum ~ rho_c a_c^2 (~p_c),
    energy ~ rho_c a_c^3. Used by the diagnostics (and, optionally, as a loss floor).
    """
    rhoL, uL, pL = 10.0 ** x[..., rhoL_idx], x[..., uL_idx], 10.0 ** x[..., pL_idx]
    rhoR, uR, pR = 10.0 ** x[..., rhoR_idx], x[..., uR_idx], 10.0 ** x[..., pR_idx]
    rho_c = 0.5 * (rhoL + rhoR)
    p_c = 0.5 * (pL + pR)
    a_c = jnp.sqrt(GAMMA * p_c / rho_c)
    v_c = a_c + jnp.maximum(jnp.abs(uL), jnp.abs(uR))
    return jnp.stack([rho_c * v_c, rho_c * v_c ** 2, rho_c * v_c ** 3], axis=-1)


def F_pred(F_net, x):
    """Network's prediction of Flux / t.
    """

    # compute flux_LR
    rhoL = 10.0**x[...,rhoL_idx]
    uL = x[...,uL_idx]
    pL = 10.0**x[...,pL_idx]
    rhoR = 10.0**x[...,rhoR_idx]
    uR = x[...,uR_idx]
    pR = 10.0**x[...,pR_idx]
    flux_L = flux_from_primitive_state(rhoL, uL, pL)
    flux_R = flux_from_primitive_state(rhoR, uR, pR)
    flux_LR = 0.5 * (flux_L + flux_R)
    
    # construct U_L and U_R
    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL**2
    U_L = jnp.array([rhoL, rhoL * uL, EL])
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR**2
    U_R = jnp.array([rhoR, rhoR * uR, ER])
    U_RL = U_R - U_L

    # compute F_net and interpret as 3x3 matrix D, then apply to U_R - U_L
    net_out = F_net(x)  # shape (9,)
    D_pred = net_out.reshape(3, 3)  # shape (3, 3)

    flux_D = D_pred @ U_RL

    flux_pred = flux_LR - 0.5 * flux_D

    return flux_pred, D_pred

def F_pred_nomat(F_net, x):
    return F_pred(F_net, x)[0]


def F_true(x: jax.Array) -> jax.Array:
    """Exact flux divided by t.
    """
    t = jnp.nan # not used for now
    gs = GasState.from_array(x)
    return compute_flux(t, gs)


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


def init_nn(in_dim: int, width: int = 32, depth: int = 3, out_dim: int = 9, *, seed: int = 0) -> MLP:
    assert out_dim == 9, "out_dim must be 9"
    dims = [in_dim] + [width] * depth + [out_dim]
    net = MLP(dims, rngs=nnx.Rngs(seed))
    net.layers[-1].kernel[...] = jnp.zeros_like(net.layers[-1].kernel[...])
    return net


def _build_model(arch: dict) -> MLP:
    """Skeleton builder for `checkpoint.load_model`. Initial weights are overwritten."""
    return init_nn(**arch, seed=0)


# ---------------------------------------------------------------------------
# Sampler + training/eval bounds

# (t, 
# log10_rhoL, uL log10_pL, 
# log10_rhoR, uR, log10_pR)
# TRAIN_BOUNDS      = jnp.array([[0., 1.0], 
#                                [-0.2, 0.2], [-0.1, 0.1], [-0.2, 0.2], 
#                                [-0.2, 0.2], [-0.1, 0.1], [-0.2, 0.2]])
# FULL_BOUNDS = TRAIN_BOUNDS
# RESTRICTED_BOUNDS = jnp.array([[0., 0.8], 
#                                 [-0.1, 0.1], [-0.05, 0.05], [-0.1, 0.1], 
#                                 [-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]])

TRAIN_BOUNDS      = jnp.array([[0., 1.0], 
                               [-2. ,2.], [-1. ,1.], [-2. ,2.], 
                               [-2. ,2.], [-1. ,1.], [-2. ,2.]])
FULL_BOUNDS = TRAIN_BOUNDS
RESTRICTED_BOUNDS = jnp.array([[0., 0.8], 
                                [-1.5, 1.5], [-0.8, 0.8], [-1.5, 1.5], 
                                [-1.5, 1.5], [-0.8, 0.8], [-1.5, 1.5]])


class UniformRandomSampler:
    """Uniform random sampler over a bounded rectangular domain."""

    def draw_batch(self, rng, batch_size: int, dim: int, bounds: jax.Array) -> jax.Array:
        lo, hi = bounds[:, 0], bounds[:, 1]
        u = jr.uniform(rng, (batch_size, dim), minval=0.0, maxval=1.0)
        return lo + (hi - lo) * u


REGIME_LABELS = ("full", "restricted", "small_jump")

# Diagnostic keys (per regime): loss concentration + zero-crossing tells.
DIAG_NAMES = (
    "top01_mass", "top01_momentum", "top01_energy",            # top-0.1% share of channel loss
    "worst1_ratio_mass", "worst1_ratio_momentum", "worst1_ratio_energy",  # median |F|/S of worst 1%
    "worst1_uabs", "all_uabs",                                  # |u_avg|: worst 1% vs all
)

METRIC_KEYS = metric_keys_for(REGIME_LABELS) + tuple(
    f"diag_{name}_{label}" for label in REGIME_LABELS for name in DIAG_NAMES
)



def make_eval_regimes(key: jax.Array, batch_size: int) -> dict[str, jax.Array]:
    k_full, k_r, k_sj = jr.split(key, 3)
    return {
        "full":       draw_rect(k_full, batch_size, FULL_BOUNDS),
        "restricted": draw_rect(k_r,    batch_size, RESTRICTED_BOUNDS),
        "small_jump": draw_small_jumps(k_sj, batch_size, RESTRICTED_BOUNDS),
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
        for name in ["mse", "same_norm"]:
            tqdm.write(f"    {name:8s}: {m[f'loss_{name}_{regime}']:.2e}")
        tqdm.write("    diagnostics (current loss):")
        for display_name, ch in (("mass", "mass"), ("momentum", "momentum"), ("energy", "energy")):
            tqdm.write(
                f"      {display_name:8s}: "
                f"top0.1%-loss={m[f'diag_top01_{ch}_{regime}']:.2f}  "
                f"worst1%_|F|/S={m[f'diag_worst1_ratio_{ch}_{regime}']:.2e}"
            )
        tqdm.write(
            f"      u@worst1%: |u|med={m[f'diag_worst1_uabs_{regime}']:.2e}  "
            f"(all={m[f'diag_all_uabs_{regime}']:.2e})"
        )
        tqdm.write("-" * 60)


# ---------------------------------------------------------------------------
# Loss

def loss_on_rel_flux_components(x: jax.Array, flux_true: jax.Array, net: MLP, lambda_mse: float, lambda_same: float, lambda_asinh: float) -> jax.Array:
    flux_pred, D_pred = jax.vmap(lambda r: F_pred(net, r))(x)
    names = ["mse", "asinh_mse", "same_norm"]

    mse = jnp.mean(jnp.abs(flux_pred - flux_true) ** 2 / (flux_scale(x))**2)

    asinh_flux_true = jnp.arcsinh(flux_true / FLUX_SCALE_ARCSINH)
    asinh_flux_pred = jnp.arcsinh(flux_pred / FLUX_SCALE_ARCSINH)
    asinh_mse = jnp.mean((asinh_flux_pred - asinh_flux_true) ** 2)

    return jnp.array([lambda_mse * mse, lambda_asinh * asinh_mse, 0.0]), names

def loss_on_rel_flux(x: jax.Array, flux_true: jax.Array, net: MLP, lambda_mse: float, lambda_same: float, lambda_asinh: float) -> jax.Array:
    components, _names = loss_on_rel_flux_components(x, flux_true, net, lambda_mse, lambda_same, lambda_asinh)
    return jnp.sum(components)

loss = loss_on_rel_flux
F_pred_fn = F_pred_nomat
loss_by_component = loss_on_rel_flux_components

def diagnostics(F_net: MLP, x: jax.Array) -> dict[str, float]:
    """Loss-concentration and zero-crossing diagnostics for one eval batch.

    Uses the flux_scale(x) denominator that the training loss optimizes, so it
    characterizes the loss actually being optimized. For each channel reports:
      - top0.1% : fraction of that channel's total loss carried by its worst
                  0.1% of samples. Near 1 => a few points dominate the loss.
      - worst1%_|F|/S : median |F_true|/S over the channel's worst 1% of
                  samples. << 1 => the worst points are near-zero-flux
                  (zero-crossings), i.e. unfittable in relative error.
    Plus |u_avg| (median) over the worst 1% by total loss vs all samples; a
    much smaller worst-value implicates u~0 zero-crossings.
    """
    flux_pred = jax.vmap(lambda r: F_pred_fn(F_net, r))(x)
    flux_true = jax.vmap(F_true)(x)
    S = flux_scale(x)

    contrib = (flux_pred - flux_true) ** 2 / S ** 2                              # (N,3)
    ratio = jnp.abs(flux_true) / S                                               # (N,3)
    u_abs = jnp.abs(0.5 * (x[..., uL_idx] + x[..., uR_idx]))                               # (N,)

    n = contrib.shape[0]
    k01 = max(1, n // 1000)
    k1 = max(1, n // 100)

    def top_share(c):
        return jnp.sort(c)[-k01:].sum() / c.sum()

    def worst_ratio(c, r):
        return jnp.median(r[jnp.argsort(c)[-k1:]])

    worst_idx = jnp.argsort(contrib.sum(axis=1))[-k1:]

    channels = ("mass", "momentum", "energy")
    out: dict[str, float] = {}
    for i, ch in enumerate(channels):
        out[f"top01_{ch}"] = float(top_share(contrib[:, i]))
        out[f"worst1_ratio_{ch}"] = float(worst_ratio(contrib[:, i], ratio[:, i]))
    out["worst1_uabs"] = float(jnp.median(u_abs[worst_idx]))
    out["all_uabs"] = float(jnp.median(u_abs))
    return out


# ---------------------------------------------------------------------------
# Training stages

@nnx.jit
def _adam_step(F_net: MLP, opt: nnx.Optimizer, batch: jax.Array, lambda_mse: float, lambda_same: float, lambda_asinh: float) -> jax.Array:
    flux_true = jax.vmap(F_true)(batch)
    loss_val, grads = nnx.value_and_grad(lambda net: loss(batch, flux_true, net, lambda_mse, lambda_same, lambda_asinh))(F_net)
    opt.update(F_net, grads)
    return loss_val


@nnx.jit
def _lbfgs_step(F_net: MLP, opt: nnx.Optimizer, x_batch: jax.Array, flux_true, lambda_mse: float, lambda_same: float, lambda_asinh: float) -> jax.Array:
    def loss_fn(m):
        return loss(x_batch, flux_true, m, lambda_mse, lambda_same, lambda_asinh)

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
    lambda_mse: float = 1e-2,
    lambda_same: float = 1e-2,
    lambda_asinh: float = 1e-2,
) -> None:
    print('lr:', lr)
    # Add gradient clipping with max_norm=1 to the optimizer
    opt = nnx.Optimizer(
        F_net,
        optax.chain(
            optax.clip_by_global_norm(1),
            optax.adamw(lr)
        ),
        wrt=nnx.Param
    )
    sampler = UniformRandomSampler()
    key = jr.PRNGKey(seed)

    pbar = tqdm(range(1, n_steps + 1), desc="adam")
    for i in pbar:
        subkey, key = jr.split(key)
        batch = sampler.draw_batch(subkey, batch_size, GAS_STATE_DIM, TRAIN_BOUNDS)
        loss_val = _adam_step(F_net, opt, batch, lambda_mse, lambda_same, lambda_asinh)
        writer.record_loss(loss_val)

        global_step = step_offset + i
        if i % 10 == 0:
            pbar.set_description(f"adam loss={float(loss_val):.4e}")
        if global_step % flush_every == 0 or i == n_steps:
            writer.flush(global_step, "adam", F_net)


def train_lbfgs(
    F_net: MLP,
    *,
    writer: CheckpointWriter,
    n_steps: int,
    step_offset: int = 0,
    flush_every: int = 100,
    batch_size: int = 2**17,
    batch_seed: int = 42,
    lambda_mse: float = 1e-2,
    lambda_same: float = 1e-2,
    lambda_asinh: float = 1e-2,
) -> None:
    sampler = UniformRandomSampler()
    x_batch = sampler.draw_batch(jr.PRNGKey(batch_seed), batch_size, GAS_STATE_DIM, TRAIN_BOUNDS)
    opt = nnx.Optimizer(F_net, optax.lbfgs(), wrt=nnx.Param)

    flux_true = jax.vmap(F_true)(x_batch)

    pbar = tqdm(range(1, n_steps + 1), desc="lbfgs")
    for i in pbar:
        loss_val = _lbfgs_step(F_net, opt, x_batch, flux_true, lambda_mse, lambda_same, lambda_asinh)
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
    return Path(__file__).resolve().parent / "ckpts/checkpoints"


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
    p.add_argument("--skip-adam", action="store_true")
    p.add_argument("--load-from-ckpt", type=Path, default=None)
    p.add_argument("--lambda-mse", type=float, default=1.)
    p.add_argument("--lambda-same", type=float, default=0.)
    p.add_argument("--lambda-asinh", type=float, default=0.)
    p.add_argument("--reset-ckpt", action="store_true")
    args = p.parse_args(argv)

    arch = {"in_dim": GAS_STATE_DIM, "width": 32, "depth": 3, "out_dim": 9}
    N_steps = 0
    if args.load_from_ckpt is not None:
        F_net, N_ = load_latest(args.load_from_ckpt, _build_model)
        N_steps += N_
    else:
        F_net = init_nn(**arch, seed=args.seed)

    eval_key = jr.PRNGKey(args.eval_seed)

    def evaluate_fn(net: nnx.Module) -> dict[str, float]:
        regimes = make_eval_regimes(eval_key, args.eval_batch_size)
        metrics = evaluate_all(
            net, F_pred_fn=F_pred_fn, F_true_fn=F_true, regimes=regimes, lambda_mse=args.lambda_mse, lambda_same=args.lambda_same, lambda_asinh=args.lambda_asinh, loss_by_component=loss_by_component,
        )
        for label, x_eval in regimes.items():
            for name, val in diagnostics(net, x_eval).items():
                metrics[f"diag_{name}_{label}"] = val
        return metrics

    writer = CheckpointWriter(
        args.ckpt_dir,
        arch,
        evaluate_fn=evaluate_fn,
        metric_keys=METRIC_KEYS,
        print_metrics_fn=print_metrics,
        reset_ckpt=args.reset_ckpt,
    )

    if not args.skip_adam:
        train_adam(
            F_net,
            writer=writer,
            n_steps=args.adam_steps,
            batch_size=args.adam_batch_size,
            step_offset=N_steps,
            flush_every=args.flush_every,
            lr=args.adam_lr,
            lambda_mse=args.lambda_mse,
            lambda_same=args.lambda_same,
            lambda_asinh=args.lambda_asinh,
        )
        N_steps += args.adam_steps

    if not args.skip_lbfgs:
        train_lbfgs(
            F_net,
            writer=writer,
            n_steps=args.lbfgs_steps,
            batch_size=args.lbfgs_batch_size,
            step_offset=N_steps,
            flush_every=args.flush_every,
            lambda_mse=args.lambda_mse,
            lambda_same=args.lambda_same,
            lambda_asinh=args.lambda_asinh,
        )
        N_steps += args.lbfgs_steps

    reloaded, _ = load_latest(args.ckpt_dir, _build_model)
    x = jr.uniform(jr.PRNGKey(7), (8, arch["in_dim"]), minval=-0.5, maxval=0.5)
    if not jnp.allclose(F_net(x), reloaded(x)):
        raise AssertionError("checkpoint round-trip mismatch")
    # print("checkpoint round-trip OK")


if __name__ == "__main__":
    main()
