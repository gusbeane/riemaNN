"""Training primitives: sampler, loss, optimizer, train loop, checkpoint, evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state as flax_train_state
from tqdm import tqdm

from . import physics
from .physics import GAS_STATE_DIM


import optax


# --- sampler ------------------------------------------------------------------


def uniform_log(
    rng,
    batch_size: int,
    *,
    log_rho_range: tuple[float, float] = (0.0, 2.0),
    log_p_range: tuple[float, float] = (-2.0, 2.0),
    u_range: tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Uniform samples in (log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)."""
    keys = jr.split(rng, 5)
    log_rho_lo, log_rho_hi = log_rho_range
    log_p_lo, log_p_hi = log_p_range
    u_lo, u_hi = u_range
    logrhoL = jr.uniform(keys[0], (batch_size,), minval=log_rho_lo, maxval=log_rho_hi)
    logpL = jr.uniform(keys[1], (batch_size,), minval=log_p_lo, maxval=log_p_hi)
    logrhoR = jr.uniform(keys[2], (batch_size,), minval=log_rho_lo, maxval=log_rho_hi)
    logpR = jr.uniform(keys[3], (batch_size,), minval=log_p_lo, maxval=log_p_hi)
    uRL = jr.uniform(keys[4], (batch_size,), minval=u_lo, maxval=u_hi)
    return jnp.stack([logrhoL, logpL, logrhoR, logpR, uRL], axis=-1)

"""
Script used to compute the quasirandom sequence golden values.

import math

def positive_root(d, tol=1e-15, max_iter=200):
    # Solve x^(d+1) = x + 1 for the unique positive root in (1, 2)
    lo, hi = 1.0, 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = mid ** (d + 1) - mid - 1.0
        if abs(f) < tol or (hi - lo) < tol:
            return mid
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

vals = [positive_root(d) for d in range(1, 13)]

print("jnp.array([")
for v in vals:
    print(f"    {v:.16f},")
print("])")
"""

def R2_quasirandom(
    rng,
    batch_size: int,
    *,
    log_rho_range: tuple[float, float],
    log_p_range: tuple[float, float],
    u_range: tuple[float, float],
) -> jnp.darray:
    """Quasirandom sequences in (log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)."""
    keys = jr.split(rng, 1)
    NDIM = 5 # currently hard-coded, but could accept more values

    gold_values = jnp.array([
        1.6180339887498949,
        1.3247179572447463,
        1.2207440846057596,
        1.1673039782614185,
        1.1347241384015194,
        1.1127756842787053,
        1.0969815577985598,
        1.0850702454914507,
        1.0757660660868371,
        1.0682971889208415,
        1.0621691678642553,
        1.0570505752212287,
    ])

    g = gold_values[NDIM - 1]
    powers = jnp.arange(1, NDIM + 1, dtype=jnp.float32)
    a = g ** (-powers)

    # allocate array and get starting pos
    # we could use a recursive relationship later, but batch_size is small so this works
    x0 = jr.uniform(rng, (NDIM,), minval=0.0, maxval=1.0)
    n = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    out_unit = jnp.mod(x0[None, :] + n * a[None, :], 1.0)

    # Scale each column to its target range
    lo = jnp.array([
        log_rho_range[0],
        log_p_range[0],
        log_rho_range[0],
        log_p_range[0],
        u_range[0],
    ], dtype=out_unit.dtype)

    hi = jnp.array([
        log_rho_range[1],
        log_p_range[1],
        log_rho_range[1],
        log_p_range[1],
        u_range[1],
    ], dtype=out_unit.dtype)

    return lo + (hi - lo) * out_unit


# --- loss ---------------------------------------------------------------------

def residual_loss_allfstar(params, apply_fn, gas_states_log):
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    return jnp.abs(fstar_vals)

def residual_loss(params, apply_fn, gas_states_log):
    """Mean squared f(p*) residual. Returns (loss, metrics)."""
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    val = fstar_vals**2
    loss = jnp.mean(val)

    return loss, {"loss/fstar": loss}

def residual_loss_normalized(params, apply_fn, gas_states_log):
    """Mean squared f(p*) residual. Returns (loss, metrics)."""
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    cref_vals = jax.vmap(physics.ref_sound_speed)(gas_states_phys)
    # pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    # val = fstar_vals**2 / pstar_true**2
    val = fstar_vals**2 / cref_vals**2
    loss = jnp.mean(val)

    return loss, {"loss/fstar": loss}

def residual_loss_newton(params, apply_fn, gas_states_log):
    """Mean squared Newton step (f/f')^2, both evaluated at NN p*.

    f/f' approximates the signed error in p*, so this is a proxy for the
    mean-squared pressure error without calling the exact solver.
    """
    pstar = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    fprime_vals = jax.vmap(physics.dfstar_dp)(pstar, gas_states_phys)
    # f' acts as a rescaling weight; don't backprop through it.
    weight = jax.lax.stop_gradient(fprime_vals)
    val = (fstar_vals / weight) ** 2
    loss = jnp.mean(val)
    return loss, {"loss/newton": loss}


def residual_loss_supervised(params, apply_fn, gas_states_log):
    pstar_NN = apply_fn({"params": params}, gas_states_log)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    pstar_true, fpstar_true = jax.vmap(physics.find_pstar)(gas_states_phys)
    loss = jnp.mean((pstar_NN - pstar_true)**2)
    return loss, {"loss": loss}


# --- optimizer ----------------------------------------------------------------


# --- train state + step -------------------------------------------------------


def create_train_state(rng, model, optimizer, batch_size_hint=256):
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )


def make_train_step(loss_fn: Callable) -> Callable:
    @jax.jit
    def train_step(state, gas_states_log):
        def _wrapped(params):
            return loss_fn(params, state.apply_fn, gas_states_log)

        (loss, metrics), grads = jax.value_and_grad(_wrapped, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics

    return train_step

def make_lbfgs_train_step(loss_fn: Callable) -> Callable:
    @jax.jit
    def train_step(state, gas_states_log):
        def value_fn(params):
            loss, _ = loss_fn(params, state.apply_fn, gas_states_log)
            return loss

        (loss, metrics), grads = jax.value_and_grad(
            lambda p: loss_fn(p, state.apply_fn, gas_states_log), has_aux=True,
        )(state.params)

        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params,
            value=loss, grad=grads, value_fn=value_fn,
        )
        new_params = optax.apply_updates(state.params, updates)

        return state.replace(
            step = state.step + 1, params=new_params, opt_state=new_opt_state
        ), loss, metrics
    
    return train_step



def run_training_loop(
    state, train_step, sampler, rng, n_epochs, batch_size,
    *, log_every=100, desc="train", split_key_every=1,
    step_callback: Callable | None = None, step_offset: int = 0,
):
    loss_trace: list[float] = []
    pbar = tqdm(range(n_epochs), desc=desc)
    for epoch in pbar:
        if (epoch % split_key_every) == 0:
            rng, batch_key = jr.split(rng)
        batch = sampler(batch_key, batch_size)
        state, loss, _metrics = train_step(state, batch)
        loss_trace.append(float(loss))
        if epoch % log_every == 0:
            pbar.set_postfix(loss=f"{loss:.2e}")
        if step_callback is not None:
            step_callback(state, step_offset + epoch + 1)

    return state, jnp.array(loss_trace)


# --- checkpoint I/O -----------------------------------------------------------


def save_checkpoint(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(state))


def load_checkpoint(path: Path, template_state):
    return from_bytes(template_state, path.read_bytes())


def save_loss_trace(path: Path, loss_trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(loss_trace))


def load_loss_trace(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)


# --- evaluation ---------------------------------------------------------------


def evaluate_holdout(state, n_samples=20_000, seed=999, **domain_kwargs):
    """Compute residual and pressure-error metrics on a holdout batch."""
    rng = jr.PRNGKey(seed)
    gas_states_log = uniform_log(rng, n_samples, **domain_kwargs)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states_phys)

    metrics: dict[str, Any] = {}

    # Residual stats
    abs_f = jnp.abs(fstar_vals)
    metrics["median_abs_fstar"] = float(jnp.median(abs_f))
    metrics["p95_abs_fstar"] = float(jnp.percentile(abs_f, 95.0))

    # Pressure error vs exact solver
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    dlogp = jnp.log10(pstar_nn) - jnp.log10(pstar_true)
    abs_dlogp = jnp.abs(dlogp)
    abs_dlogp_np = np.asarray(abs_dlogp)
    metrics["median_abs_delta_log10_p"] = float(np.nanmedian(abs_dlogp_np))
    metrics["p95_abs_delta_log10_p"] = float(np.nanpercentile(abs_dlogp_np, 95.0))

    abs_absolute = jnp.abs(pstar_nn - pstar_true)
    metrics["abs_absolute_median"] = float(np.nanmedian(abs_absolute))
    metrics["abs_absolute_p95"] = float(np.nanpercentile(abs_absolute, 95.0))
    metrics["abs_absolute_p5"] = float(np.nanpercentile(abs_absolute, 5.0))

    any_nan_nn = bool(jnp.any(jnp.isnan(pstar_nn)))
    any_nan_true = bool(jnp.any(jnp.isnan(pstar_true)))
    metrics["any_nan_nn"] = "true" if any_nan_nn else "false"
    metrics["any_nan_true"] = "true" if any_nan_true else "false"

    any_neg_nn = bool(jnp.any(pstar_nn < 0))
    any_neg_true = bool(jnp.any(pstar_true < 0))
    metrics["any_neg_nn"] = "true" if any_neg_nn else "false"
    metrics["any_neg_true"] = "true" if any_neg_true else "false"

    return metrics
