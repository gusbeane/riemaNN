"""Train a face-flux network for the gamma = 5/3 Euler Riemann problem.

The pipeline is sampler -> predictor -> loss, each a small swappable piece
selected by name on the CLI and assembled in `main()`:
  --sampler   : where training points come from            (SAMPLERS)
  --predictor : how raw net output becomes a flux          (PREDICTORS)
  --loss      : how flux error becomes named components    (LOSSES)

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

# Make sibling modules importable when invoked as `python innt/riemann/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import CheckpointWriter, load_latest  # noqa: E402
from evaluate import (  # noqa: E402
    draw_rect,
    draw_small_jumps,
    evaluate_all,
    metric_keys_for,
)
from physics import GAMMA, GAS_STATE_DIM, GasState, compute_flux, flux_from_primitive_state

FLUX_SCALE_ARCSINH = jnp.array([1e-2, 1e-3, 2e-3])


def flux_scale(x: jax.Array) -> jax.Array:
    """Per-sample, per-channel characteristic Euler flux scale (mass, momentum, energy).

    Built from the average state: mass ~ rho_c a_c, momentum ~ rho_c a_c^2 (~p_c),
    energy ~ rho_c a_c^3. Used by the diagnostics (and, optionally, as a loss floor).
    """
    gs = GasState.from_array(x)
    rho_c = 0.5 * (gs.rhoL + gs.rhoR)
    p_c = 0.5 * (gs.pL + gs.pR)
    a_c = jnp.sqrt(GAMMA * p_c / rho_c)
    v_c = a_c + jnp.maximum(jnp.abs(gs.uL), jnp.abs(gs.uR))
    return jnp.stack([rho_c * v_c, rho_c * v_c ** 2, rho_c * v_c ** 3], axis=-1)


def F_true(x: jax.Array) -> jax.Array:
    """Exact interface flux from the exact Riemann solution."""
    t = jnp.nan  # compute_flux is self-similar; t is unused
    return compute_flux(t, GasState.from_array(x))


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


def init_nn(in_dim: int, width: int = 32, depth: int = 3, out_dim: int = 9, *,
            zero_init_last: bool = False, seed: int = 0) -> MLP:
    dims = [in_dim] + [width] * depth + [out_dim]
    net = MLP(dims, rngs=nnx.Rngs(seed))
    if zero_init_last:
        net.layers[-1].kernel[...] = jnp.zeros_like(net.layers[-1].kernel[...])
    return net


# ---------------------------------------------------------------------------
# Predictors: how raw network output is interpreted as a flux.
#
# A predictor owns the facts that must stay mutually consistent: the network's
# `out_dim`, any init special-casing (`zero_init_last`), and the map from net
# output to flux. `__call__` takes the net and a single sample x of shape
# (GAS_STATE_DIM,) and returns `(flux, aux)`; callers vmap. `aux` is a dict of
# parameterization-specific extras (e.g. the dissipation matrix) for losses
# and diagnostics that want them.

class RoeMatrixPredictor:
    """flux = 0.5 (F_L + F_R) - 0.5 D (U_R - U_L), with D = net(x) as 3x3."""

    out_dim = 9
    zero_init_last = True  # start at the average-flux baseline (D = 0)

    def __call__(self, net, x):
        gs = GasState.from_array(x)
        rhoL, uL, pL = gs.rhoL, gs.uL, gs.pL
        rhoR, uR, pR = gs.rhoR, gs.uR, gs.pR

        flux_L = flux_from_primitive_state(rhoL, uL, pL)
        flux_R = flux_from_primitive_state(rhoR, uR, pR)
        flux_LR = 0.5 * (flux_L + flux_R)

        EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL**2
        ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR**2
        U_L = jnp.array([rhoL, rhoL * uL, EL])
        U_R = jnp.array([rhoR, rhoR * uR, ER])
        U_RL = U_R - U_L

        D = net(x).reshape(3, 3)
        flux = flux_LR - 0.5 * (D @ U_RL)
        return flux, {"D": D}


# A star-state predictor (net outputs e.g. (p*, u*, rho*) -> flux) slots in
# here with its own out_dim.
PREDICTORS = {
    "roe_mat": RoeMatrixPredictor(),
}


def _build_model(arch: dict, *, seed: int = 0) -> MLP:
    """Skeleton builder for `checkpoint.load_model`. Initial weights are overwritten."""
    predictor = PREDICTORS[arch["predictor"]]
    return init_nn(
        arch["in_dim"], width=arch["width"], depth=arch["depth"],
        out_dim=predictor.out_dim, zero_init_last=predictor.zero_init_last, seed=seed,
    )


# ---------------------------------------------------------------------------
# Samplers + training/eval bounds

# (log10_rhoL, uL, log10_pL, log10_rhoR, uR, log10_pR)
TRAIN_BOUNDS      = jnp.array([[-2., 2.], [-1., 1.], [-2., 2.],
                               [-2., 2.], [-1., 1.], [-2., 2.]])
FULL_BOUNDS = TRAIN_BOUNDS
RESTRICTED_BOUNDS = jnp.array([[-1.5, 1.5], [-0.8, 0.8], [-1.5, 1.5],
                               [-1.5, 1.5], [-0.8, 0.8], [-1.5, 1.5]])


class UniformRandomSampler:
    """Uniform random sampler over a bounded rectangular domain."""

    def __init__(self, bounds: jax.Array):
        self.bounds = bounds

    def draw_batch(self, key: jax.Array, batch_size: int) -> jax.Array:
        return draw_rect(key, batch_size, self.bounds)


# Values are classes; main() constructs the chosen one with TRAIN_BOUNDS.
SAMPLERS = {
    "uniform": UniformRandomSampler,
}


REGIME_LABELS = ("full", "restricted", "small_jump")


def make_eval_regimes(key: jax.Array, batch_size: int) -> dict[str, jax.Array]:
    k_full, k_r, k_sj = jr.split(key, 3)
    return {
        "full":       draw_rect(k_full, batch_size, FULL_BOUNDS),
        "restricted": draw_rect(k_r,    batch_size, RESTRICTED_BOUNDS),
        "small_jump": draw_small_jumps(k_sj, batch_size, RESTRICTED_BOUNDS),
    }


# ---------------------------------------------------------------------------
# Losses: map (x, flux_pred, flux_true, aux) -> dict of named, unweighted
# scalar components. The training loss is sum(lambdas[name] * components[name])
# with `lambdas` built once in main() from the --lambda-* flags; adding a
# component means adding a dict entry here and a matching flag.

def rel_flux_components(x: jax.Array, flux_pred: jax.Array, flux_true: jax.Array,
                        aux: dict) -> dict[str, jax.Array]:
    mse = jnp.mean(jnp.abs(flux_pred - flux_true) ** 2 / flux_scale(x) ** 2)

    asinh_flux_true = jnp.arcsinh(flux_true / FLUX_SCALE_ARCSINH)
    asinh_flux_pred = jnp.arcsinh(flux_pred / FLUX_SCALE_ARCSINH)
    asinh_mse = jnp.mean((asinh_flux_pred - asinh_flux_true) ** 2)

    return {"mse": mse, "asinh_mse": asinh_mse}


# name -> (components_fn, component names). The names are listed explicitly so
# metric keys / CSV headers exist before any loss is evaluated.
LOSSES = {
    "rel_flux": (rel_flux_components, ("mse", "asinh_mse")),
}


def make_train_loss(predictor, loss_components, lambdas: dict[str, float]):
    """Bind predictor + loss + weights into `loss(net, x, flux_true) -> scalar`."""

    def train_loss(net, x: jax.Array, flux_true: jax.Array) -> jax.Array:
        flux_pred, aux = jax.vmap(lambda r: predictor(net, r))(x)
        components = loss_components(x, flux_pred, flux_true, aux)
        return sum(lambdas[name] * components[name] for name in lambdas)

    return train_loss


# ---------------------------------------------------------------------------
# Diagnostics + metric printing

# Diagnostic keys (per regime): loss concentration + zero-crossing tells.
DIAG_NAMES = (
    "top01_mass", "top01_momentum", "top01_energy",            # top-0.1% share of channel loss
    "worst1_ratio_mass", "worst1_ratio_momentum", "worst1_ratio_energy",  # median |F|/S of worst 1%
    "worst1_uabs", "all_uabs",                                  # |u_avg|: worst 1% vs all
)


def diagnostics(predictor, F_net: MLP, x: jax.Array) -> dict[str, float]:
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
    flux_pred, _aux = jax.vmap(lambda r: predictor(F_net, r))(x)
    flux_true = jax.vmap(F_true)(x)
    S = flux_scale(x)
    gs = GasState.from_array(x)

    contrib = (flux_pred - flux_true) ** 2 / S ** 2                              # (N,3)
    ratio = jnp.abs(flux_true) / S                                               # (N,3)
    u_abs = jnp.abs(0.5 * (gs.uL + gs.uR))                                       # (N,)

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


def print_metrics(m: Mapping[str, float], loss_names: tuple[str, ...]) -> None:
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
        for name in loss_names:
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
# Training stages

def make_adam_step(train_loss):
    @nnx.jit
    def step(F_net: MLP, opt: nnx.Optimizer, batch: jax.Array) -> jax.Array:
        flux_true = jax.vmap(F_true)(batch)
        loss_val, grads = nnx.value_and_grad(lambda m: train_loss(m, batch, flux_true))(F_net)
        opt.update(F_net, grads)
        return loss_val

    return step


def make_lbfgs_step(train_loss):
    @nnx.jit
    def step(F_net: MLP, opt: nnx.Optimizer, x_batch: jax.Array, flux_true: jax.Array) -> jax.Array:
        def loss_fn(m):
            return train_loss(m, x_batch, flux_true)

        graphdef, _params, rest = nnx.split(F_net, nnx.Param, ...)
        loss_val, grads = nnx.value_and_grad(loss_fn)(F_net)

        def value_fn(trial_params):
            return loss_fn(nnx.merge(graphdef, trial_params, rest))

        opt.update(F_net, grads, value=loss_val, grad=grads, value_fn=value_fn)
        return loss_val

    return step


def train_adam(
    F_net: MLP,
    *,
    step_fn,
    sampler,
    writer: CheckpointWriter,
    n_steps: int,
    step_offset: int = 0,
    flush_every: int = 100,
    lr: float = 1e-3,
    batch_size: int = 100_000,
    seed: int = 0,
) -> None:
    print('lr:', lr)
    opt = nnx.Optimizer(
        F_net,
        optax.chain(
            optax.clip_by_global_norm(1),
            optax.adamw(lr)
        ),
        wrt=nnx.Param
    )
    key = jr.PRNGKey(seed)

    pbar = tqdm(range(1, n_steps + 1), desc="adam")
    for i in pbar:
        subkey, key = jr.split(key)
        batch = sampler.draw_batch(subkey, batch_size)
        loss_val = step_fn(F_net, opt, batch)
        writer.record_loss(loss_val)

        global_step = step_offset + i
        if i % 10 == 0:
            pbar.set_description(f"adam loss={float(loss_val):.4e}")
        if global_step % flush_every == 0 or i == n_steps:
            writer.flush(global_step, "adam", F_net)


def train_lbfgs(
    F_net: MLP,
    *,
    step_fn,
    sampler,
    writer: CheckpointWriter,
    n_steps: int,
    step_offset: int = 0,
    flush_every: int = 100,
    batch_size: int = 2**17,
    batch_seed: int = 42,
) -> None:
    x_batch = sampler.draw_batch(jr.PRNGKey(batch_seed), batch_size)
    opt = nnx.Optimizer(F_net, optax.lbfgs(), wrt=nnx.Param)

    flux_true = jax.vmap(F_true)(x_batch)

    pbar = tqdm(range(1, n_steps + 1), desc="lbfgs")
    for i in pbar:
        loss_val = step_fn(F_net, opt, x_batch, flux_true)
        writer.record_loss(loss_val)

        global_step = step_offset + i
        if i % 10 == 0:
            pbar.set_description(f"lbfgs loss={float(loss_val):.4e}")
        if global_step % flush_every == 0 or i == n_steps:
            writer.flush(global_step, "lbfgs", F_net)


# ---------------------------------------------------------------------------
# CLI

def _default_ckpt_dir() -> Path:
    return Path(__file__).resolve().parent / "ckpts/checkpoints"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, default=_default_ckpt_dir())
    p.add_argument("--predictor", choices=sorted(PREDICTORS), default="roe_mat")
    p.add_argument("--loss", choices=sorted(LOSSES), default="rel_flux")
    p.add_argument("--sampler", choices=sorted(SAMPLERS), default="uniform")
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
    p.add_argument("--lambda-asinh", type=float, default=0.)
    p.add_argument("--reset-ckpt", action="store_true")
    args = p.parse_args(argv)

    predictor = PREDICTORS[args.predictor]
    loss_components, loss_names = LOSSES[args.loss]
    sampler = SAMPLERS[args.sampler](TRAIN_BOUNDS)

    lambda_flags = {"mse": args.lambda_mse, "asinh_mse": args.lambda_asinh}
    lambdas = {name: lambda_flags[name] for name in loss_names}

    train_loss = make_train_loss(predictor, loss_components, lambdas)
    adam_step = make_adam_step(train_loss)
    lbfgs_step = make_lbfgs_step(train_loss)

    # `predictor` is part of the checkpoint arch: out_dim and init are derived
    # from it on reload, so a checkpoint can't be silently reinterpreted under
    # a different parameterization.
    arch = {"in_dim": GAS_STATE_DIM, "width": 32, "depth": 3, "predictor": args.predictor}
    N_steps = 0
    if args.load_from_ckpt is not None:
        F_net, N_ = load_latest(args.load_from_ckpt, _build_model)
        N_steps += N_
    else:
        F_net = _build_model(arch, seed=args.seed)

    if F_net.layers[-1].kernel[...].shape[-1] != predictor.out_dim:
        raise ValueError(
            f"loaded network out_dim {F_net.layers[-1].kernel[...].shape[-1]} "
            f"does not match predictor {args.predictor!r} (out_dim {predictor.out_dim})"
        )

    eval_key = jr.PRNGKey(args.eval_seed)

    def evaluate_fn(net: nnx.Module) -> dict[str, float]:
        regimes = make_eval_regimes(eval_key, args.eval_batch_size)
        metrics = evaluate_all(
            net,
            F_pred_fn=lambda m, r: predictor(m, r)[0],
            F_true_fn=F_true,
            regimes=regimes,
        )
        for label, x_eval in regimes.items():
            flux_pred, aux = jax.vmap(lambda r: predictor(net, r))(x_eval)
            flux_true = jax.vmap(F_true)(x_eval)
            components = loss_components(x_eval, flux_pred, flux_true, aux)
            for name in loss_names:
                metrics[f"loss_{name}_{label}"] = float(components[name])
            for name, val in diagnostics(predictor, net, x_eval).items():
                metrics[f"diag_{name}_{label}"] = val
        return metrics

    metric_keys = (
        metric_keys_for(REGIME_LABELS)
        + tuple(f"loss_{name}_{label}" for label in REGIME_LABELS for name in loss_names)
        + tuple(f"diag_{name}_{label}" for label in REGIME_LABELS for name in DIAG_NAMES)
    )

    writer = CheckpointWriter(
        args.ckpt_dir,
        arch,
        evaluate_fn=evaluate_fn,
        metric_keys=metric_keys,
        print_metrics_fn=lambda m: print_metrics(m, loss_names),
        reset_ckpt=args.reset_ckpt,
    )

    if not args.skip_adam:
        train_adam(
            F_net,
            step_fn=adam_step,
            sampler=sampler,
            writer=writer,
            n_steps=args.adam_steps,
            batch_size=args.adam_batch_size,
            step_offset=N_steps,
            flush_every=args.flush_every,
            lr=args.adam_lr,
        )
        N_steps += args.adam_steps

    if not args.skip_lbfgs:
        train_lbfgs(
            F_net,
            step_fn=lbfgs_step,
            sampler=sampler,
            writer=writer,
            n_steps=args.lbfgs_steps,
            batch_size=args.lbfgs_batch_size,
            step_offset=N_steps,
            flush_every=args.flush_every,
        )
        N_steps += args.lbfgs_steps

    reloaded, _ = load_latest(args.ckpt_dir, _build_model)
    x = jr.uniform(jr.PRNGKey(7), (8, arch["in_dim"]), minval=-0.5, maxval=0.5)
    if not jnp.allclose(F_net(x), reloaded(x)):
        raise AssertionError("checkpoint round-trip mismatch")


if __name__ == "__main__":
    main()
