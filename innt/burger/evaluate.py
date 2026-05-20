"""L2/Linf error evaluation against a known ground truth + sampling helpers
for the evaluation regimes used by the Burgers integrated face-flux pipeline.

The error machinery (`errors_on`, `evaluate_all`) is physics-agnostic: the
caller supplies
- `F_pred_fn(F_net, x_scalar)`  -> scalar prediction for a single input,
- `F_true_fn(x_scalar)`         -> scalar ground truth for a single input,
- `regimes: dict[str, Array]`   -> mapping from a regime label to a batch of
                                   evaluation points of shape `(N, dim)`.

The sampling helpers (`draw_rect`, `draw_small_jumps`) currently encode the
Burgers `(t, uL, uR)` regimes used to build that dict. They live here for
convenience; a later refactor can split the evaluation regimes out per
physics setup.
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
) -> tuple[float, float]:
    """Return (L2, Linf) error of `F_pred_fn(F_net, .)` against `F_true_fn` on `x_eval`."""
    F_pred = jax.vmap(lambda r: F_pred_fn(F_net, r))(x_eval)
    F_exact = jax.vmap(F_true_fn)(x_eval)
    err = F_pred - F_exact
    l2 = float(jnp.sqrt(jnp.mean(err**2)))
    linf = float(jnp.max(jnp.abs(err)))
    return l2, linf


def evaluate_all(
    F_net,
    *,
    F_pred_fn: Callable[..., jax.Array],
    F_true_fn: Callable[[jax.Array], jax.Array],
    regimes: Mapping[str, jax.Array],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label, x_eval in regimes.items():
        l2, linf = errors_on(F_net, x_eval, F_pred_fn, F_true_fn)
        metrics[f"l2_{label}"] = l2
        metrics[f"linf_{label}"] = linf
    return metrics


def metric_keys_for(regime_labels) -> tuple[str, ...]:
    """Return the metric-key tuple produced by `evaluate_all` for these regimes,
    matching the (l2, linf) per-regime order used internally."""
    return tuple(f"{m}_{label}" for label in regime_labels for m in ("l2", "linf"))


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
    *,
    u_min: float = 0.1,
    u_max: float = 0.8,
    max_ratio: float = 1e-2,
    t_max: float = 1.0,
) -> jax.Array:
    """Batch of `(t, uL, uR)` with `|u_avg| in [u_min, u_max]` and `|du/u| <= max_ratio`.

    Burgers / Riemann-shock-tube-shaped: returns triples where the left/right
    states straddle a target average magnitude by a controllable small jump.
    """
    k_t, k_mag, k_sign, k_r = jr.split(key, 4)
    t = jr.uniform(k_t, (batch_size,), minval=0.0, maxval=t_max)
    u_mag = jr.uniform(k_mag, (batch_size,), minval=u_min, maxval=u_max)
    u_sign = jnp.where(jr.uniform(k_sign, (batch_size,)) < 0.5, -1.0, 1.0)
    u_avg = u_sign * u_mag
    delta_u = jr.uniform(k_r, (batch_size,), minval=-max_ratio, maxval=max_ratio) * u_mag
    uL = u_avg - 0.5 * delta_u
    uR = u_avg + 0.5 * delta_u
    return jnp.stack([t, uL, uR], axis=-1)
