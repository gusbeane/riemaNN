"""Riemann problem physics for an ideal gas with gamma = 5/3.

Provides the jump functions from the exact Riemann solver (Toro, ch. 4),
a JAX Newton iteration for the star pressure p*, and helpers for moving
between log and linear representations of the left/right gas state.

A "gas state" is a length-5 array `(rhoL, pL, rhoR, pR, uRL)` where
`uRL = uR - uL` is the velocity difference across the interface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

GAMMA: float = 5.0 / 3.0
ALPHA: float = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA: float = (GAMMA - 1.0) / (GAMMA + 1.0)
MU: float = (GAMMA - 1.0) / 2.0

GAS_STATE_DIM: int = 5

@jax.jit
def sound_speed(p, rho):
    return jnp.sqrt(GAMMA*p/rho)

@jax.jit
def ref_sound_speed(gas_state):
    rhoL, pL, rhoR, pR, _uRL = gas_state
    cL = sound_speed(pL, rhoL)
    cR = sound_speed(pR, rhoR)
    return 0.5*(cL + cR)

@jax.jit
def fjump(p, pK, rhoK):
    """Contribution of one side (K = L or R) to the Riemann star-pressure equation."""
    AK = 2.0 / ((GAMMA + 1.0) * rhoK)
    BK = pK * BETA
    cK = jnp.sqrt(GAMMA * pK / rhoK)
    shock = jnp.sqrt(AK / (p + BK)) * (p - pK)
    rarefaction = ((p / pK) ** ALPHA - 1.0) * cK / MU
    return jnp.where(p > pK, shock, rarefaction)


@jax.jit
def fstar(pstar, gas_state):
    """Residual of the Riemann star-pressure equation; zero at the true p*."""
    rhoL, pL, rhoR, pR, uRL = gas_state
    return fjump(pstar, pL, rhoL) + fjump(pstar, pR, rhoR) + uRL


dfstar_dp = jax.grad(fstar, argnums=0)


@jax.jit
def _newton(gas_state, p0):
    """Newton iteration for p*. Fast but can diverge."""

    def cond(state):
        pstar, pstar_prev, fstar_, i = state
        return (jnp.abs(fstar_) >= 1e-12) & (jnp.abs(pstar - pstar_prev) >= 1e-10) & (i < 100)

    def body(state):
        pstar, _pstar_prev, _, i = state
        pstar_prev = pstar
        fstar_ = fstar(pstar, gas_state)
        dfstar_ = dfstar_dp(pstar, gas_state)
        pstar = pstar - fstar_ / dfstar_
        return pstar, pstar_prev, fstar(pstar, gas_state), i + 1

    # state = (pguess, pguess_prev, fstar, i)
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

    # --- 1. Bracket p* around the guess ---
    p_lo = p_guess * 0.5
    p_hi = p_guess * 2.0
    f_lo = fstar(p_lo, gas_state)
    f_hi = fstar(p_hi, gas_state)

    # Widen until signs differ or we hit the iteration limit.
    # fstar is monotonically increasing, so we lower p_lo when
    # f_lo > 0 and raise p_hi when f_hi < 0.
    def widen_cond(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        return (_f_lo * _f_hi > 0) & (i < 60)

    def widen_body(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        _p_lo = jnp.where(_f_lo > 0, jnp.maximum(_p_lo * 0.01, 1e-30), _p_lo)
        _p_hi = jnp.where(_f_hi < 0, _p_hi * 100.0, _p_hi)
        return (_p_lo, _p_hi,
                fstar(_p_lo, gas_state), fstar(_p_hi, gas_state), i + 1)

    p_lo, p_hi, f_lo, f_hi, _ = jax.lax.while_loop(
        widen_cond, widen_body, (p_lo, p_hi, f_lo, f_hi, 0)
    )

    # --- 2–3. Bisect to convergence ---
    def bisect_cond(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        return ((_p_hi - _p_lo) > 1e-12 * jnp.maximum(p_mid, 1e-30)) & (i < 200)

    def bisect_body(state):
        _p_lo, _p_hi, i = state
        p_mid = 0.5 * (_p_lo + _p_hi)
        f_mid = fstar(p_mid, gas_state)
        # fstar is monotonically increasing: f < 0 means p too low
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
    """Find p* via Newton with bisection fallback; returns (pstar, residual).

    Tries Newton iteration first (fast, ~5-10 steps).  If Newton fails
    to converge or produces a non-finite / non-positive result, falls
    back to the guaranteed-convergent bisection solver.
    """
    p0 = jnp.maximum(two_rarefaction_p0(gas_state), 1e-30)
    p_newton, f_newton = _newton(gas_state, p0)

    newton_ok = (jnp.isfinite(p_newton) & jnp.isfinite(f_newton)
                 & (p_newton > 0) & (jnp.abs(f_newton) < 1e-12))

    p_bisect, f_bisect = jax.lax.cond(
        newton_ok,
        lambda _: (p_newton, f_newton),
        lambda _: _bisect(gas_state),
        None,
    )
    return p_bisect, f_bisect


@jax.jit
def find_ustar(gas_state, pstar):
    """Star-region velocity u* = 0.5 * (uRL + fR(p*) - fL(p*)) in the uRL=uR-uL convention.

    Useful for experiments that also train on u* (e.g. STAR_PRESSURE_VELOCITY target).
    """
    rhoL, pL, rhoR, pR, _ = gas_state
    fL = fjump(pstar, pL, rhoL)
    fR = fjump(pstar, pR, rhoR)
    return 0.5 * (fR - fL)


@jax.jit
def two_rarefaction_p0(gas_state):
    """Two-rarefaction approximation for p* (Toro eq. 4.46)."""
    rhoL, pL, rhoR, pR, uRL = gas_state
    cL = jnp.sqrt(GAMMA * pL / rhoL)
    cR = jnp.sqrt(GAMMA * pR / rhoR)
    numerator = jnp.maximum(cL + cR - MU * uRL, 1e-30)
    denominator = cL / pL**ALPHA + cR / pR**ALPHA
    return (numerator / denominator) ** (1.0 / ALPHA)


two_rarefaction_p0_batch = jax.vmap(two_rarefaction_p0)


def gas_log_to_phys(gas_states_log):
    """(batch, 5) log10(rhoL,pL,rhoR,pR) + uRL -> physical (rhoL,pL,rhoR,pR,uRL)."""
    return jnp.concatenate(
        [10.0 ** gas_states_log[:, :4], gas_states_log[:, 4:5]], axis=-1
    )


def gas_phys_to_log(gas_states_phys):
    """(batch, 5) physical (rhoL,pL,rhoR,pR,uRL) -> log10(rhoL,pL,rhoR,pR) + uRL."""
    return jnp.concatenate(
        [jnp.log10(gas_states_phys[:, :4]), gas_states_phys[:, 4:5]], axis=-1
    )


