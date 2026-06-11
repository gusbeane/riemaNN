"""Per-channel relative-error evaluation against a known ground truth + sampling
helpers for the evaluation regimes used by the Riemann integrated face-flux
pipeline.

The error machinery (`errors_on`, `evaluate_all`) is physics-agnostic: the
caller supplies
- `F_pred_fn(F_net, x_scalar)`  -> prediction for a single input (scalar or
                                   vector; reduction is over all axes),
- `F_true_fn(x_scalar)`         -> ground truth for a single input,
- `regimes: dict[str, Array]`   -> mapping from a regime label to a batch of
                                   evaluation points of shape `(N, dim)`.

The sampling helpers (`draw_rect`, `draw_small_jumps`) encode the Riemann
`(log10_rhoL, uL, log10_pL, log10_rhoR, uR, log10_pR)` regimes used to build
that dict.
"""

from __future__ import annotations

from typing import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import random as jr


def errors_on(
    F_net,
    x_eval: jax.Array,
    F_pred_fn: Callable[..., jax.Array],
    F_true_fn: Callable[[jax.Array], jax.Array],
    rel_tol=1e-2,
) -> tuple[jax.Array, jax.Array]:
    """Per-channel relative-error stats of `F_pred_fn(F_net, .)` vs `F_true_fn`.

    The relative error is `|F_pred - F_true| / |F_true|`, reduced over the batch
    axis. Returns `(median_rel_channels, max_rel_channels)`, each of shape
    `(n_channels,)` = (mass, momentum, energy).
    """
    F_pred = jax.vmap(lambda r: F_pred_fn(F_net, r))(x_eval)
    F_exact = jax.vmap(F_true_fn)(x_eval)
    rel = jnp.where(
        jnp.abs(F_exact) > rel_tol,
        jnp.abs(F_pred - F_exact) / jnp.abs(F_exact),
        jnp.nan,
    )

    median_rel_channels = jnp.nanmedian(rel, axis=0)
    max_rel_channels = jnp.nanmax(rel, axis=0)
    return median_rel_channels, max_rel_channels


def evaluate_all(
    F_net,
    *,
    F_pred_fn: Callable[..., jax.Array],
    F_true_fn: Callable[[jax.Array], jax.Array],
    regimes: Mapping[str, jax.Array],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label, x_eval in regimes.items():
        median_rel_channels, max_rel_channels = errors_on(
            F_net, x_eval, F_pred_fn, F_true_fn
        )
        for i, channel in enumerate(CHANNEL_LABELS):
            metrics[f"median_rel_{label}_{channel}"] = float(median_rel_channels[i])
            metrics[f"max_rel_{label}_{channel}"] = float(max_rel_channels[i])

    return metrics


CHANNEL_LABELS = ("mass_flux", "momentum_flux", "energy_flux")


def metric_keys_for(regime_labels) -> tuple[str, ...]:
    """Return the metric-key tuple produced by `evaluate_all` for these regimes,
    matching the per-channel (median_rel, max_rel) order used internally."""
    return tuple(
        f"{m}_{label}_{channel}"
        for label in regime_labels
        for channel in CHANNEL_LABELS
        for m in ("median_rel", "max_rel")
    )


# ---------------------------------------------------------------------------
# Sampling helpers for evaluation regimes

def draw_rect(key: jax.Array, batch_size: int, bounds: jax.Array) -> jax.Array:
    """Uniform random samples in a rectangular box.

    `bounds` has shape `(dim, 2)`, each row `[lo, hi]`.
    Returns an `(batch_size, dim)` array.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    u = jr.uniform(key, (batch_size, bounds.shape[0]), minval=0.0, maxval=1.0)
    return lo + (hi - lo) * u


def draw_small_jumps(
    key: jax.Array,
    batch_size: int,
    bounds: jax.Array,
    *,
    jump_max_rel: float = 1e-3,
) -> jax.Array:
    """Batch of states whose R components are relative jumps of at most
    `jump_max_rel` off the L components.

    Near-constant-state regime: small perturbations of a uniform fluid where
    the exact Riemann fan reduces to linear acoustics. Useful for checking
    the network in the weak-jump limit, which the full uniform sampler covers
    with vanishing probability.
    """
    k_j, k_r = jr.split(key, 2)
    # Draw the L state (log10_rhoL, uL, log10_pL) from the first three bounds
    # rows, then a relative jump r in [-jump_max_rel, jump_max_rel] per
    # component and set R = L * (1 + r) in linear variables.
    l_lo = bounds[[0, 1, 2], 0]
    l_hi = bounds[[0, 1, 2], 1]
    u_L = jr.uniform(k_j, (batch_size, 3), minval=0.0, maxval=1.0)
    L_vals = l_lo + (l_hi - l_lo) * u_L  # shape (batch_size, 3)

    # Convert the log columns (rho, p) to linear before applying the jump.
    L_vals_lin = L_vals.at[:, [0, 2]].set(10.0 ** L_vals[:, [0, 2]])

    r = jr.uniform(k_r, (batch_size, 3), minval=-jump_max_rel, maxval=jump_max_rel)

    R_vals_lin = L_vals_lin * (1.0 + r)
    R_vals = R_vals_lin.at[:, [0, 2]].set(jnp.log10(R_vals_lin[:, [0, 2]]))

    return jnp.concatenate([L_vals, R_vals], axis=-1)
