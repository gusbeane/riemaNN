"""Loss implementations parameterized over a Target.

A loss function has the signature
    (target, preds, gas_states_log, gas_states_phys, **kwargs) -> (loss, metrics)
where `preds` is the dict returned by `target.decode`, `metrics` is a
dict of named scalar floats for logging, and `loss` is a scalar jax array.

`make_loss_fn` wraps a loss implementation + target into a closure
`(params, apply_fn, gas_states_log) -> (loss, metrics)` suitable for
`jax.value_and_grad` and the training loop.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from . import physics
from .targets import Target

LossFn = Callable[..., tuple[jnp.ndarray, dict[str, jnp.ndarray]]]


def residual_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Mean squared physical residual. Matches current 1-ideal_riemann.py behavior.

    `weights` is an optional per-residual-name float weight; defaults to 1.0
    for every residual returned by `target.residuals`.
    """
    residuals = target.residuals(preds, gas_states_phys)
    weights = weights or {}
    total = jnp.zeros((), dtype=jnp.float64)
    metrics: dict[str, jnp.ndarray] = {}
    for name, r in residuals.items():
        w = weights.get(name, 1.0)
        component = jnp.mean(r ** 2)
        total = total + w * component
        metrics[f"loss/{name}"] = component
    metrics["loss/total"] = total
    return total, metrics


def supervised_loss(*args, **kwargs):
    """MSE against Newton solution. Implement when a supervised experiment is needed."""
    raise NotImplementedError(
        "supervised_loss is not yet implemented. Add it when the first "
        "supervised experiment is needed; the signature should mirror "
        "residual_loss but take a `reference: Callable` kwarg that maps "
        "gas_states_phys -> ground-truth predictions dict."
    )


def combined_loss(*args, **kwargs):
    """Weighted sum of residual_loss and supervised_loss."""
    raise NotImplementedError(
        "combined_loss is not yet implemented. Implement alongside supervised_loss."
    )


def make_loss_fn(target: Target, loss_impl: LossFn, **loss_kwargs: Any) -> Callable:
    """Return a closure `(params, apply_fn, gas_states_log) -> (loss, metrics)`.

    Captures `target`, `loss_impl`, and any loss hyperparameters so the
    training loop can call the result uniformly.
    """

    def loss_fn(params, apply_fn, gas_states_log):
        raw = apply_fn({"params": params}, gas_states_log)
        preds = target.decode(gas_states_log, raw)
        gas_states_phys = physics.gas_log_to_phys(gas_states_log)
        loss, metrics = loss_impl(
            target, preds, gas_states_log, gas_states_phys, **loss_kwargs
        )
        return loss, metrics

    return loss_fn
