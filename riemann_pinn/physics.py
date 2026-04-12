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
def find_pstar(gas_state, p0=None):
    """Newton iteration for p*; returns (pstar, final residual).

    If p0 is None (default), the two-rarefaction approximation is used
    as the initial guess, which converges for a much wider range of
    input states than the old fixed p0=1.0 default.
    """
    if p0 is None:
        p0 = two_rarefaction_p0(gas_state)

    def cond(state):
        pstar, fstar_, i = state
        return (jnp.abs(fstar_) >= 1e-6) & (i < 100)

    def body(state):
        pstar, _, i = state
        fstar_ = fstar(pstar, gas_state)
        dfstar_ = dfstar_dp(pstar, gas_state)
        pstar = pstar - fstar_ / dfstar_
        return pstar, fstar(pstar, gas_state), i + 1

    init = (p0, fstar(p0, gas_state), 0)
    pstar, fstar_final, _ = jax.lax.while_loop(cond, body, init)
    return pstar, fstar_final


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
