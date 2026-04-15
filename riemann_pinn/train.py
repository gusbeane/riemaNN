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


# --- loss ---------------------------------------------------------------------


def residual_loss(params, apply_fn, gas_states_log):
    """Mean squared f(p*) residual. Returns (loss, metrics)."""
    raw = apply_fn({"params": params}, gas_states_log)
    pstar = 10.0 ** raw
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    fstar_vals = jax.vmap(physics.fstar)(pstar, gas_states_phys)
    loss = jnp.mean(fstar_vals ** 2)
    return loss, {"loss/fstar": loss}

def residual_loss_supervised(params, apply_fn, gas_states_log):
    raw = apply_fn({"params": params}, gas_states_log)
    pstar_NN = 10.0 ** raw
    # gas_states_phys = jax.vmap(physics.gas_log_to_phys)(gas_states_log)
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


def run_training_loop(
    state, train_step, sampler, rng, n_epochs, batch_size,
    *, log_every=100, desc="train",
):
    loss_trace: list[float] = []
    pbar = tqdm(range(n_epochs), desc=desc)
    for epoch in pbar:
        rng, batch_key = jr.split(rng)
        batch = sampler(batch_key, batch_size)
        state, loss, _metrics = train_step(state, batch)
        loss_trace.append(float(loss))
        if epoch % log_every == 0:
            pbar.set_postfix(loss=f"{loss:.2e}")
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

    raw_out = state.apply_fn({"params": state.params}, gas_states_log)
    pstar_nn = 10.0 ** raw_out
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states_phys)

    metrics: dict[str, Any] = {}

    # Residual stats
    abs_f = jnp.abs(fstar_vals)
    metrics["median_abs_fstar"] = float(jnp.median(abs_f))
    metrics["p95_abs_fstar"] = float(jnp.percentile(abs_f, 95.0))

    # Newton-step metric: |f/f'| approximates |delta p|
    dfstar_vals = jax.vmap(physics.dfstar_dp)(pstar_nn, gas_states_phys)
    safe_df = jnp.where(
        jnp.abs(dfstar_vals) > 1e-20, dfstar_vals,
        jnp.sign(dfstar_vals + 1e-30) * 1e-20,
    )
    newton_steps = jnp.abs(fstar_vals / safe_df)
    metrics["median_abs_newton_step"] = float(jnp.median(newton_steps))
    metrics["p95_abs_newton_step"] = float(jnp.percentile(newton_steps, 95.0))

    # Pressure error vs exact solver
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    pos = (pstar_true > 1e-30) & (pstar_nn > 1e-30)
    dlogp = jnp.log10(pstar_nn) - jnp.log10(pstar_true)
    abs_dlogp = jnp.abs(dlogp)
    abs_dlogp = jnp.where(pos, abs_dlogp, jnp.nan)
    abs_dlogp_np = np.asarray(abs_dlogp)
    metrics["median_abs_delta_log10_p"] = float(np.nanmedian(abs_dlogp_np))
    metrics["p95_abs_delta_log10_p"] = float(np.nanpercentile(abs_dlogp_np, 95.0))
    metrics["frac_both_p_positive"] = float(jnp.mean(pos))

    metrics["frac_worse_50percent"] = float(
        jnp.mean((dlogp > jnp.log10(1.5)) | (dlogp < jnp.log10(0.5)))
    )
    
    metrics["frac_worse_1percent"] = float(
        jnp.mean((dlogp > jnp.log10(1.01)) | (dlogp < jnp.log10(0.99)))
    )

    metrics["frac_worse_0p1percent"] = float(
        jnp.mean((dlogp > jnp.log10(1.001)) | (dlogp < jnp.log10(0.999)))
    )

    rel_p = jnp.abs(pstar_nn - pstar_true) / jnp.maximum(jnp.abs(pstar_true), 1e-30)
    metrics["median_rel_abs_p_err"] = float(jnp.median(rel_p))
    metrics["p95_rel_abs_p_err"] = float(jnp.percentile(rel_p, 95.0))

    return metrics
