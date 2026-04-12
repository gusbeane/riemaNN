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


def newton_step_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    newton_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Combined residual + Newton-step loss.

    Includes both mean(f^2) and mean((f/f')^2).  Using (f/f')^2 alone
    has a degenerate solution where the network makes f' -> infinity;
    the residual term prevents that.
    """
    residuals = target.residuals(preds, gas_states_phys)
    metrics: dict[str, jnp.ndarray] = {}

    fstar_vals = residuals["fstar"]
    res_component = jnp.mean(fstar_vals ** 2)
    metrics["loss/residual_fstar"] = res_component

    pstar = preds["pstar"]
    dfstar_vals = jax.vmap(physics.dfstar_dp)(pstar, gas_states_phys)
    safe_df = jnp.where(jnp.abs(dfstar_vals) > 1e-20, dfstar_vals,
                        jnp.sign(dfstar_vals + 1e-30) * 1e-20)
    newton_steps = fstar_vals / safe_df
    newton_component = jnp.mean(newton_steps ** 2)
    metrics["loss/newton_step_fstar"] = newton_component

    total = residual_weight * res_component + newton_weight * newton_component
    metrics["loss/total"] = total
    return total, metrics


def symmetry_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    params,
    apply_fn,
    raw,
    symmetry_weight: float = 0.1,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Residual loss + symmetry penalty (augmented signature).

    Symmetry term: mean((raw - raw_swapped)^2) where the swap exchanges
    L/R states and flips the sign of uRL.
    """
    residuals = target.residuals(preds, gas_states_phys)
    res_loss = jnp.mean(residuals["fstar"] ** 2)

    gas_swap = jnp.concatenate([
        gas_states_log[:, 2:4],
        gas_states_log[:, 0:2],
        -gas_states_log[:, 4:5],
    ], axis=-1)
    raw_swapped = apply_fn({"params": params}, gas_swap)
    sym_loss = jnp.mean((raw - raw_swapped) ** 2)

    total = res_loss + symmetry_weight * sym_loss
    metrics = {
        "loss/residual_fstar": res_loss,
        "loss/symmetry": sym_loss,
        "loss/total": total,
    }
    return total, metrics


def derivative_loss(
    target: Target,
    preds: dict[str, jnp.ndarray],
    gas_states_log: jnp.ndarray,
    gas_states_phys: jnp.ndarray,
    *,
    params,
    apply_fn,
    raw,
    deriv_uRL_weight: float = 0.0,
    deriv_rho_weight: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Residual loss + derivative penalties (augmented signature).

    Compares the model's input-gradients against analytical derivatives
    of the true p*.  Targets are evaluated at the Newton-solved p* (not
    the predicted p*) for a stable training signal.

    Log-space analytical targets:
      d(log10 p*)/d(uRL)       = -1 / (p_true * ln(10) * D)
      d(log10 p*)/d(log10 rhoK) = fK(p_true) / (2 * D * p_true)
    where D = dfstar/dp evaluated at the true p*.
    """
    residuals = target.residuals(preds, gas_states_phys)
    res_loss = jnp.mean(residuals["fstar"] ** 2)

    metrics: dict[str, jnp.ndarray] = {"loss/residual_fstar": res_loss}
    total = res_loss

    need_grads = (deriv_uRL_weight > 0.0) or (deriv_rho_weight > 0.0)
    if need_grads:
        # True p* from Newton iteration
        p_true, fstar_residual = jax.vmap(physics.find_pstar)(gas_states_phys)
        D = jax.vmap(physics.dfstar_dp)(p_true, gas_states_phys)

        # Mask out samples where Newton didn't converge or p* <= 0
        valid = (p_true > 1e-30) & jnp.isfinite(D) & (jnp.abs(D) > 1e-20)
        n_valid = jnp.maximum(jnp.sum(valid), 1)

        safe_D = jnp.where(valid, D, 1.0)  # placeholder for masked samples
        safe_p = jnp.where(valid, p_true, 1.0)

        # Model Jacobian: d(raw)/d(input) per sample
        def model_scalar(x_single):
            return apply_fn({"params": params}, x_single[None, :])[0]
        model_grads = jax.vmap(jax.grad(model_scalar))(gas_states_log)

    if deriv_uRL_weight > 0.0:
        analytical_duRL = -1.0 / (safe_p * jnp.log(10.0) * safe_D)
        model_duRL = model_grads[:, 4]
        duRL_err = jnp.where(valid, (model_duRL - analytical_duRL) ** 2, 0.0)
        duRL_loss = jnp.sum(duRL_err) / n_valid
        total = total + deriv_uRL_weight * duRL_loss
        metrics["loss/deriv_uRL"] = duRL_loss

    if deriv_rho_weight > 0.0:
        rhoL, pL, rhoR, pR, _ = [gas_states_phys[:, i] for i in range(5)]
        fL = jax.vmap(physics.fjump)(safe_p, pL, rhoL)
        fR = jax.vmap(physics.fjump)(safe_p, pR, rhoR)

        analytical_drhoL = fL / (2.0 * safe_D * safe_p)
        analytical_drhoR = fR / (2.0 * safe_D * safe_p)

        model_drhoL = model_grads[:, 0]
        model_drhoR = model_grads[:, 2]

        drho_errL = jnp.where(valid, (model_drhoL - analytical_drhoL) ** 2, 0.0)
        drho_errR = jnp.where(valid, (model_drhoR - analytical_drhoR) ** 2, 0.0)
        drho_loss = 0.5 * (jnp.sum(drho_errL) / n_valid
                           + jnp.sum(drho_errR) / n_valid)
        total = total + deriv_rho_weight * drho_loss
        metrics["loss/deriv_rho"] = drho_loss

    metrics["loss/total"] = total
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


def make_augmented_loss_fn(
    target: Target, loss_impl: LossFn, **loss_kwargs: Any
) -> Callable:
    """Like make_loss_fn, but additionally passes params, apply_fn, and raw to loss_impl.

    Use this for loss implementations that need to re-evaluate the model
    (symmetry_loss) or compute input gradients (derivative_loss).
    """

    def loss_fn(params, apply_fn, gas_states_log):
        raw = apply_fn({"params": params}, gas_states_log)
        preds = target.decode(gas_states_log, raw)
        gas_states_phys = physics.gas_log_to_phys(gas_states_log)
        loss, metrics = loss_impl(
            target,
            preds,
            gas_states_log,
            gas_states_phys,
            params=params,
            apply_fn=apply_fn,
            raw=raw,
            **loss_kwargs,
        )
        return loss, metrics

    return loss_fn
