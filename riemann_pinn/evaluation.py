"""Holdout metrics and slice-grid helpers.

Evaluation is generic over a Target — `evaluate_holdout` computes
percentiles for every residual the target returns, plus pressure-error
metrics (|Δlog10 p|, relative error) whenever the target exposes a
`pstar` key in its decoded predictions.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from . import physics
from .samplers import uniform_log
from .targets import Target


def evaluate_holdout(
    state,
    target: Target,
    n_samples: int = 20_000,
    seed: int = 999,
) -> dict[str, Any]:
    """Compute residual and (when applicable) pressure-error metrics on a holdout batch."""
    rng = jr.PRNGKey(seed)
    gas_states_log = uniform_log(rng, n_samples)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)

    raw_out = state.apply_fn({"params": state.params}, gas_states_log)
    preds = target.decode(gas_states_log, raw_out)
    residuals = target.residuals(preds, gas_states_phys)

    metrics: dict[str, Any] = {}
    for name, r in residuals.items():
        abs_r = jnp.abs(r)
        metrics[f"median_abs_{name}"] = float(jnp.median(abs_r))
        metrics[f"p95_abs_{name}"] = float(jnp.percentile(abs_r, 95.0))

    # Newton-step metric: |f/f'| approximates |delta p|
    if "fstar" in residuals and "pstar" in preds:
        dfstar_vals = jax.vmap(physics.dfstar_dp)(preds["pstar"], gas_states_phys)
        safe_df = jnp.where(jnp.abs(dfstar_vals) > 1e-20, dfstar_vals,
                            jnp.sign(dfstar_vals + 1e-30) * 1e-20)
        newton_steps = jnp.abs(residuals["fstar"] / safe_df)
        metrics["median_abs_newton_step"] = float(jnp.median(newton_steps))
        metrics["p95_abs_newton_step"] = float(jnp.percentile(newton_steps, 95.0))

    # Pressure-error metrics require a pstar prediction and the exact solver.
    if "pstar" in preds:
        pstar_nn = preds["pstar"]
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)

        pos = (pstar_true > 1e-30) & (pstar_nn > 1e-30)
        abs_dlogp = jnp.abs(jnp.log10(pstar_nn) - jnp.log10(pstar_true))
        abs_dlogp = jnp.where(pos, abs_dlogp, jnp.nan)
        abs_dlogp_np = np.asarray(abs_dlogp)
        metrics["median_abs_delta_log10_p"] = float(np.nanmedian(abs_dlogp_np))
        metrics["p95_abs_delta_log10_p"] = float(np.nanpercentile(abs_dlogp_np, 95.0))
        metrics["frac_both_p_positive"] = float(jnp.mean(pos))

        rel_p = jnp.abs(pstar_nn - pstar_true) / jnp.maximum(
            jnp.abs(pstar_true), 1e-30
        )
        metrics["median_rel_abs_p_err"] = float(jnp.median(rel_p))
        metrics["p95_rel_abs_p_err"] = float(jnp.percentile(rel_p, 95.0))

    return metrics


def holdout_log_ratio(
    state,
    target: Target,
    n_samples: int = 20_000,
    seed: int = 999,
) -> np.ndarray:
    """Compute log10(p_NN / p_true) on a holdout batch. Returns (n_samples,) numpy array."""
    rng = jr.PRNGKey(seed)
    gas_states_log = uniform_log(rng, n_samples)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)

    raw_out = state.apply_fn({"params": state.params}, gas_states_log)
    preds = target.decode(gas_states_log, raw_out)
    pstar_nn = preds["pstar"]
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)

    log_ratio = jnp.log10(jnp.maximum(pstar_nn, 1e-30)) - jnp.log10(jnp.maximum(pstar_true, 1e-30))
    return np.asarray(log_ratio)


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Pretty one-line summary for console output."""
    parts = [prefix] if prefix else []
    if "median_abs_fstar" in metrics:
        parts.append(
            f"|f|: median={metrics['median_abs_fstar']:.3e}, "
            f"p95={metrics['p95_abs_fstar']:.3e}"
        )
    if "median_abs_delta_log10_p" in metrics:
        parts.append(
            f"|Δlog10 p| (both p>0, frac={metrics['frac_both_p_positive']:.2f}): "
            f"median={metrics['median_abs_delta_log10_p']:.3e}, "
            f"p95={metrics['p95_abs_delta_log10_p']:.3e}"
        )
    if "median_rel_abs_p_err" in metrics:
        parts.append(
            f"|p_NN-p_true|/|p_true|: median={metrics['median_rel_abs_p_err']:.3e}, "
            f"p95={metrics['p95_rel_abs_p_err']:.3e}"
        )
    return "  |  ".join(parts)


# --- slice grid ---------------------------------------------------------------


def slice_grid_data(
    n: int = 100,
    log_rho_range: tuple[float, float] = (-2.0, 2.0),
    log_p_range: tuple[float, float] = (-2.0, 2.0),
    logrhoR_fix: float = 0.0,
    logpR_fix: float = 0.0,
    uRL_fix: float = 0.0,
):
    """Build a 2D slice over (log rhoL, log pL) with the right state and uRL fixed."""
    logrhoL_vals = jnp.linspace(log_rho_range[0], log_rho_range[1], n)
    logpL_vals = jnp.linspace(log_p_range[0], log_p_range[1], n)
    logrhoL_grid, logpL_grid = jnp.meshgrid(logrhoL_vals, logpL_vals, indexing="ij")
    gas_states_log = jnp.stack(
        [
            logrhoL_grid.ravel(),
            logpL_grid.ravel(),
            jnp.full(n * n, logrhoR_fix),
            jnp.full(n * n, logpR_fix),
            jnp.full(n * n, uRL_fix),
        ],
        axis=-1,
    )
    return gas_states_log, logrhoL_vals, logpL_vals, n


def compute_slice_fields(state, target: Target, gas_states_log, n: int) -> dict[str, np.ndarray]:
    """Evaluate the slice grid. Returns a dict of shape-(n,n) numpy arrays.

    Keys (when target exposes `pstar`):
      - `log_ratio`: log10(p_NN / p_true)
      - `signed_log_abs_fstar`: sign(f) * log10(max(|f|, 1e-30))
    Additional residuals in target.residuals(...) are also logged.
    """
    raw_out = state.apply_fn({"params": state.params}, gas_states_log)
    preds = target.decode(gas_states_log, raw_out)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    residuals = target.residuals(preds, gas_states_phys)

    out: dict[str, np.ndarray] = {}

    if "pstar" in preds:
        pstar_nn = preds["pstar"]
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
        log_ratio = jnp.log10(pstar_nn / pstar_true).reshape(n, n)
        out["log_ratio"] = np.asarray(log_ratio)

    if "fstar" in residuals:
        f = residuals["fstar"]
        signed = (
            jnp.sign(f) * jnp.log10(jnp.maximum(jnp.abs(f), 1e-30))
        ).reshape(n, n)
        out["signed_log_abs_fstar"] = np.asarray(signed)

    return out
