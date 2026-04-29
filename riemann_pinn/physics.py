"""Riemann problem physics for an ideal gas with gamma = 5/3, 3D form.

A "gas state" is a length-3 array `(drho, dp, du)`:

  drho = (rhoR - rhoL) / (rhoR + rhoL)  in [-1, 1]
  dp   = (pR   - pL)   / (pR   + pL)    in [-1, 1]
  du   = uRL / ducrit(drho, dp)         in [-inf, 1]

The non-dimensionalization p_ref = 1, rho_ref = 1 is implicit:
pL = 1 - dp, pR = 1 + dp, rhoL = 1 - drho, rhoR = 1 + drho.
Sound speeds, ducrit, and p* are all dimensionless (ratios to p_ref).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

GAMMA: float = 5.0 / 3.0
ALPHA: float = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA: float = (GAMMA - 1.0) / (GAMMA + 1.0)
MU: float = (GAMMA - 1.0) / 2.0

GAS_STATE_DIM: int = 3


@jax.jit
def get_ducrit(drho, dp):
    """Returns the speed of the vacuum solution relative to the reference sound speed.
    Note that c_ref = sqrt(gamma * p_ref/rho_ref).
    """
    ansL = jnp.sqrt((1 + dp) / (1 + drho))
    ansR = jnp.sqrt((1 - dp) / (1 - drho))
    return (2.0 / (GAMMA - 1.0)) * (ansL + ansR)


@jax.jit
def ftilde(p, drho, dp, LR):
    """Contribution of one side (K = L or R) to the Riemann star-pressure equation.

    LR = -1 for L, +1 for R.
    """
    AK = jnp.sqrt(2.0 / (GAMMA * (GAMMA + 1.0) * (1 + LR * drho)))
    BK = BETA * (1 + LR * dp)

    shock = (p - (1 + LR * dp)) * AK / jnp.sqrt(p + BK)
    rarefaction = (1.0 / MU) * jnp.sqrt(
        (1 + LR * dp) / (1 + LR * drho)
    ) * ((p / (1 + LR * dp)) ** ALPHA - 1.0)
    return jnp.where(p > (1 + LR * dp), shock, rarefaction)


@jax.jit
def fstar(p, gas_state):
    """Residual of the Riemann star-pressure equation; zero at the true p*."""
    drho, dp, du = gas_state
    ducrit = get_ducrit(drho, dp)
    return (ftilde(p, drho, dp, -1) + ftilde(p, drho, dp, 1)) / ducrit + du


dfstar_dp = jax.grad(fstar, argnums=0)


@jax.jit
def two_rarefaction_p0(gas_state):
    """3D two-rarefaction p* guess (Toro eq. 4.46), dimensionless.

    Used only as the Newton / bisection starting point inside find_pstar.
    """
    drho, dp, du = gas_state
    cL = jnp.sqrt(GAMMA * (1 - dp) / (1 - drho))
    cR = jnp.sqrt(GAMMA * (1 + dp) / (1 + drho))
    ducrit = get_ducrit(drho, dp)
    num = jnp.maximum(cL + cR - MU * du * ducrit, 1e-30)
    den = cL / (1 - dp) ** ALPHA + cR / (1 + dp) ** ALPHA
    return (num / den) ** (1.0 / ALPHA)


@jax.jit
def _newton(gas_state, p0):
    """Newton iteration for p*. Fast but can diverge."""

    def cond(state):
        pstar, pstar_prev, fstar_, i = state
        return (
            (jnp.abs(fstar_) >= 1e-12)
            & (jnp.abs(pstar - pstar_prev) >= 1e-10)
            & (i < 100)
        )

    def body(state):
        pstar, _pstar_prev, _, i = state
        pstar_prev = pstar
        fstar_ = fstar(pstar, gas_state)
        dfstar_ = dfstar_dp(pstar, gas_state)
        pstar = pstar - fstar_ / dfstar_
        return pstar, pstar_prev, fstar(pstar, gas_state), i + 1

    init = (p0, jnp.inf, fstar(p0, gas_state), 0)
    pstar, _pstar_prev, fstar_final, _ = jax.lax.while_loop(cond, body, init)
    return pstar, fstar_final


@jax.jit
def _bisect(gas_state):
    """Bisection solver for p*. Slow but guaranteed to converge.

    Vacuum states (no physical root) return a very small p* with a
    non-zero residual.
    """
    p_guess = jnp.maximum(two_rarefaction_p0(gas_state), 1e-30)

    p_lo = p_guess * 0.5
    p_hi = p_guess * 2.0
    f_lo = fstar(p_lo, gas_state)
    f_hi = fstar(p_hi, gas_state)

    def widen_cond(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        return (_f_lo * _f_hi > 0) & (i < 60)

    def widen_body(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        _p_lo = jnp.where(_f_lo > 0, jnp.maximum(_p_lo * 0.01, 1e-30), _p_lo)
        _p_hi = jnp.where(_f_hi < 0, _p_hi * 100.0, _p_hi)
        return (
            _p_lo, _p_hi,
            fstar(_p_lo, gas_state), fstar(_p_hi, gas_state), i + 1,
        )

    p_lo, p_hi, f_lo, f_hi, _ = jax.lax.while_loop(
        widen_cond, widen_body, (p_lo, p_hi, f_lo, f_hi, 0)
    )

    def bisect_cond(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        return ((_p_hi - _p_lo) > 1e-12 * jnp.maximum(p_mid, 1e-30)) & (i < 200)

    def bisect_body(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        f_mid = fstar(p_mid, gas_state)
        _p_lo = jnp.where(f_mid < 0, p_mid, _p_lo)
        _p_hi = jnp.where(f_mid >= 0, p_mid, _p_hi)
        return _p_lo, _p_hi, i + 1

    p_lo, p_hi, _ = jax.lax.while_loop(
        bisect_cond, bisect_body, (p_lo, p_hi, 0)
    )

    p_result = 0.5 * (p_lo + p_hi)
    f_result = fstar(p_result, gas_state)
    return p_result, f_result


@jax.jit
def find_pstar(gas_state):
    """Find p* via Newton with bisection fallback; returns (pstar, residual)."""
    p0 = jnp.maximum(two_rarefaction_p0(gas_state), 1e-30)
    p_newton, f_newton = _newton(gas_state, p0)

    newton_ok = (
        jnp.isfinite(p_newton)
        & jnp.isfinite(f_newton)
        & (p_newton > 0)
        & (jnp.abs(f_newton) < 1e-12)
    )

    p_bisect, f_bisect = jax.lax.cond(
        newton_ok,
        lambda _: (p_newton, f_newton),
        lambda _: _bisect(gas_state),
        None,
    )
    return p_bisect, f_bisect
