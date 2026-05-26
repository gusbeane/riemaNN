"""L2/Linf error evaluation against a known ground truth + sampling helpers
for the evaluation regimes used by the Riemann integrated face-flux pipeline.

The error machinery (`errors_on`, `evaluate_all`) is physics-agnostic: the
caller supplies
- `F_pred_fn(F_net, x_scalar)`  -> prediction for a single input (scalar or
                                   vector; reduction is over all axes),
- `F_true_fn(x_scalar)`         -> ground truth for a single input,
- `regimes: dict[str, Array]`   -> mapping from a regime label to a batch of
                                   evaluation points of shape `(N, dim)`.

The sampling helpers (`draw_rect`, `draw_small_jumps`) encode the Riemann
`(t, drho, dp, du)` regimes used to build that dict.
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
    jump_max: float = 1e-2,
    t_max: float = 1.0,
) -> jax.Array:
    """Batch of `(t, drho, dp, du)` with `|drho|, |dp|, |du| <= jump_max`.

    Near-constant-state regime: small perturbations of a uniform fluid where
    the exact Riemann fan reduces to linear acoustics. Useful for checking
    the network in the weak-jump limit, which the full uniform sampler covers
    with vanishing probability.
    """
    k_t, k_j = jr.split(key, 2)
    t = jr.uniform(k_t, (batch_size,), minval=0.0, maxval=t_max)
    jumps = jr.uniform(k_j, (batch_size, 3), minval=-jump_max, maxval=jump_max)
    return jnp.concatenate([t[:, None], jumps], axis=-1)
