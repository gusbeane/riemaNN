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

import jax
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


def huber_residual_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    delta: float = 1.0,
    weights: dict[str, float] | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Huber penalty on physical residuals. Tames the hard tail of residual_loss.

    For |r| <= delta: 0.5 * r^2 (quadratic); for |r| > delta: delta * (|r| - 0.5 * delta)
    (linear). This keeps gradients bounded for outlier samples instead of letting
    them dominate the step direction.
    """
    residuals = target.residuals(preds, gas_states_phys)
    weights = weights or {}
    total = jnp.zeros((), dtype=jnp.float64)
    metrics: dict[str, jnp.ndarray] = {}
    for name, r in residuals.items():
        w = weights.get(name, 1.0)
        abs_r = jnp.abs(r)
        quadratic = jnp.minimum(abs_r, delta)
        linear = abs_r - quadratic
        component = jnp.mean(0.5 * quadratic ** 2 + delta * linear)
        total = total + w * component
        metrics[f"loss/huber_{name}"] = component
    metrics["loss/total"] = total
    return total, metrics


def _default_pstar_reference(gas_states_phys: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Default ground-truth for supervised training: Newton-iterated p* via physics.find_pstar."""
    pstar, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    return {"pstar": pstar}


def supervised_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    reference: Callable[[jnp.ndarray], dict[str, jnp.ndarray]] = _default_pstar_reference,
    in_log_space: bool = True,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MSE against a reference solution (default: Newton-solved p*).

    - `reference` maps `gas_states_phys` -> dict of ground-truth predictions with
      the same keys as `preds`. Default calls `vmap(physics.find_pstar)`.
    - `in_log_space=True` compares `log10(pred)` to `log10(truth)`, which is
      appropriate for the p* target (wide dynamic range). Samples where the
      reference is non-positive (Newton divergence) are masked out.
    - `in_log_space=False` does linear MSE on the raw predictions.
    """
    truth = reference(gas_states_phys)
    total = jnp.zeros((), dtype=jnp.float64)
    metrics: dict[str, jnp.ndarray] = {}
    for name, pred in preds.items():
        if name not in truth:
            continue
        true = truth[name]
        if in_log_space:
            safe_pred = jnp.maximum(pred, 1e-30)
            safe_true = jnp.maximum(true, 1e-30)
            err_sq = (jnp.log10(safe_pred) - jnp.log10(safe_true)) ** 2
            valid = true > 1e-30
            err_sq_masked = jnp.where(valid, err_sq, 0.0)
            n_valid = jnp.maximum(jnp.sum(valid), 1)
            component = jnp.sum(err_sq_masked) / n_valid
        else:
            component = jnp.mean((pred - true) ** 2)
        total = total + component
        metrics[f"loss/supervised_{name}"] = component
    metrics["loss/total"] = total
    return total, metrics


def combined_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    residual_weight: float = 1.0,
    supervised_weight: float = 1.0,
    reference: Callable[[jnp.ndarray], dict[str, jnp.ndarray]] = _default_pstar_reference,
    in_log_space: bool = True,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Weighted sum of residual_loss and supervised_loss."""
    r_loss, r_metrics = residual_loss(target, preds, gas_states_log, gas_states_phys)
    s_loss, s_metrics = supervised_loss(
        target,
        preds,
        gas_states_log,
        gas_states_phys,
        reference=reference,
        in_log_space=in_log_space,
    )
    total = residual_weight * r_loss + supervised_weight * s_loss
    metrics = {**r_metrics, **s_metrics, "loss/total": total}
    return total, metrics


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
