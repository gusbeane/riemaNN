"""Riemann problem physics for an ideal gas with gamma = 5/3, 3D form.

A "gas state" is a length-3 array `(drho, dp, du)`:

  drho = (rhoR - rhoL) / (rhoR + rhoL)  in [-1, 1]
  dp   = (pR   - pL)   / (pR   + pL)    in [-1, 1]
  du   = uRL / ducrit(drho, dp)         in [-inf, 1]
  
  where uRL = uR - uL

The non-dimensionalization p_ref = 1, rho_ref = 1 is implicit:
pL = 1 - dp, pR = 1 + dp, rhoL = 1 - drho, rhoR = 1 + drho.
Sound speeds, ducrit, and p* are all dimensionless (ratios to p_ref).
So c_ref**2 = p_ref / rho_ref.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple

GAMMA: float = 5.0 / 3.0
ALPHA: float = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA: float = (GAMMA - 1.0) / (GAMMA + 1.0)
MU: float = (GAMMA - 1.0) / 2.0

GAS_STATE_DIM: int = 3

class GasState(NamedTuple):
    rhoL: jax.Array
    pL:   jax.Array
    rhoR: jax.Array
    pR:   jax.Array
    uRL:  jax.Array

    @classmethod
    def from_array(cls, x):
        return cls(
            rhoL=x[..., 0],
            pL=x[..., 1],
            rhoR=x[..., 2],
            pR=x[..., 3],
            uRL=x[..., 4],
        )
    
    def as_array(self):
        return jnp.stack([self.rhoL, self.pL, self.rhoR, self.pR, self.uRL], axis=-1)
    
    @property
    def aL(self):
        return jnp.sqrt(GAMMA * self.pL / self.rhoL)
    
    @property
    def aR(self):
        return jnp.sqrt(GAMMA * self.pR / self.rhoR)

@jax.jit
def get_ducrit(drho, dp):
    """Returns the speed of the vacuum solution relative to the reference sound speed.
    Note that c_ref = sqrt(p_ref/rho_ref).
    """
    ansL = jnp.sqrt((1 + dp) / (1 + drho))
    ansR = jnp.sqrt((1 - dp) / (1 - drho))
    return (2.0 * jnp.sqrt(GAMMA) / (GAMMA - 1.0)) * (ansL + ansR)


@jax.jit
def ftilde(p, drho, dp, LR):
    """Contribution of one side (K = L or R) to the Riemann star-pressure equation.

    LR = -1 for L, +1 for R.
    """
    AK = jnp.sqrt(2.0 / ((GAMMA + 1.0) * (1 + LR * drho)))
    BK = BETA * (1 + LR * dp)

    shock = (p - (1 + LR * dp)) * AK / jnp.sqrt(p + BK)
    rarefaction = (1.0 / MU) * jnp.sqrt(
        GAMMA * (1 + LR * dp) / (1 + LR * drho)
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
def two_rarefaction_p0(gs: GasState):
    """3D two-rarefaction p* guess (Toro eq. 4.46), dimensionless.

    Used only as the Newton / bisection starting point inside find_pstar.
    """
    aL = gs.aL ; aR = gs.aR

    num = aL + aR - MU * gs.uRL
    den = aL / gs.pL ** ALPHA + aR / gs.pR ** ALPHA
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
def find_pstar(gs: GasState):
    """Find p* via Newton with bisection fallback; returns (pstar, residual)."""
    p0 = jnp.maximum(two_rarefaction_p0(gs), 1e-30)
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


@jax.jit
def sample_origin(gas_state, pstar, uL=0.0):
    """Return rho, u, e at x=0, t>0.

    Velocities are in units of u_ref = sqrt(p_ref / rho_ref).
    Pressure and density use p_ref = rho_ref = 1 normalization.

    uL fixes the Galilean frame. If uL=0, then uR = du * ducrit.
    """
    drho, dp, du = gas_state

    rhoL = 1.0 - drho
    rhoR = 1.0 + drho
    pL = 1.0 - dp
    pR = 1.0 + dp

    ducrit = get_ducrit(drho, dp)
    uR = uL + du * ducrit

    # Sound speeds divided by c_ref.
    aL = jnp.sqrt(GAMMA * pL / rhoL)
    aR = jnp.sqrt(GAMMA * pR / rhoR)

    fL = ftilde(pstar, drho, dp, -1)
    fR = ftilde(pstar, drho, dp, +1)

    ustar_L = uL - fL
    ustar_R = uR + fR
    ustar = 0.5 * (ustar_L + ustar_R)

    xi = 0.0

    def star_density(rhoK, pK):
        pratio = pstar / pK

        rho_shock = rhoK * (pratio + BETA) / (BETA * pratio + 1.0)
        rho_raref = rhoK * pratio ** (1.0 / GAMMA)

        return jnp.where(pstar > pK, rho_shock, rho_raref)

    rhoL_star = star_density(rhoL, pL)
    rhoR_star = star_density(rhoR, pR)

    eL = pL / ((GAMMA - 1.0) * rhoL)
    eR = pR / ((GAMMA - 1.0) * rhoR)
    eL_star = pstar / ((GAMMA - 1.0) * rhoL_star)
    eR_star = pstar / ((GAMMA - 1.0) * rhoR_star)

    def left_fan_state():
        # Left rarefaction fan state at xi = 0.
        u = (2.0 / (GAMMA + 1.0)) * (
            aL + MU * uL + xi
        )
        a = (2.0 / (GAMMA + 1.0)) * (
            aL + MU * (uL - xi)
        )
        rho = rhoL * (a / aL) ** (2.0 / (GAMMA - 1.0))
        p = pL * (a / aL) ** (2.0 * GAMMA / (GAMMA - 1.0))
        e = p / ((GAMMA - 1.0) * rho)
        return rho, u, e

    def right_fan_state():
        # Right rarefaction fan state at xi = 0.
        u = (2.0 / (GAMMA + 1.0)) * (
            -aR + MU * uR + xi
        )
        a = (2.0 / (GAMMA + 1.0)) * (
            aR - MU * (uR - xi)
        )
        rho = rhoR * (a / aR) ** (2.0 / (GAMMA - 1.0))
        p = pR * (a / aR) ** (2.0 * GAMMA / (GAMMA - 1.0))
        e = p / ((GAMMA - 1.0) * rho)
        return rho, u, e

    def sample_left_of_contact():
        # We are left of the contact: possible states are L, L*, or left fan.
        is_shock = pstar > pL

        def shock_case():
            SL = uL - aL * jnp.sqrt(
                (GAMMA + 1.0) / (2.0 * GAMMA) * (pstar / pL)
                + (GAMMA - 1.0) / (2.0 * GAMMA)
            )

            # For a left shock:
            # xi <= SL gives original L; xi > SL gives L*.
            return jax.lax.cond(
                xi <= SL,
                lambda _: (rhoL, uL, eL),
                lambda _: (rhoL_star, ustar, eL_star),
                None,
            )

        def raref_case():
            SHL = uL - aL
            aL_star = aL * (pstar / pL) ** ALPHA
            STL = ustar - aL_star

            # Left rarefaction:
            # xi <= SHL: L
            # SHL < xi < STL: fan
            # xi >= STL: L*
            return jax.lax.cond(
                xi <= SHL,
                lambda _: (rhoL, uL, eL),
                lambda _: jax.lax.cond(
                    xi >= STL,
                    lambda __: (rhoL_star, ustar, eL_star),
                    lambda __: left_fan_state(),
                    None,
                ),
                None,
            )

        return jax.lax.cond(is_shock, lambda _: shock_case(), lambda _: raref_case(), None)

    def sample_right_of_contact():
        # We are right of the contact: possible states are R*, R, or right fan.
        is_shock = pstar > pR

        def shock_case():
            SR = uR + aR * jnp.sqrt(
                (GAMMA + 1.0) / (2.0 * GAMMA) * (pstar / pR)
                + (GAMMA - 1.0) / (2.0 * GAMMA)
            )

            # For a right shock:
            # xi <= SR gives R*; xi > SR gives original R.
            return jax.lax.cond(
                xi <= SR,
                lambda _: (rhoR_star, ustar, eR_star),
                lambda _: (rhoR, uR, eR),
                None,
            )

        def raref_case():
            SHR = uR + aR
            aR_star = aR * (pstar / pR) ** ALPHA
            STR = ustar + aR_star

            # Right rarefaction:
            # xi <= STR: R*
            # STR < xi < SHR: fan
            # xi >= SHR: R
            return jax.lax.cond(
                xi <= STR,
                lambda _: (rhoR_star, ustar, eR_star),
                lambda _: jax.lax.cond(
                    xi >= SHR,
                    lambda __: (rhoR, uR, eR),
                    lambda __: right_fan_state(),
                    None,
                ),
                None,
            )

        return jax.lax.cond(is_shock, lambda _: shock_case(), lambda _: raref_case(), None)

    # Contact speed separates left-star and right-star sides.
    return jax.lax.cond(
        xi <= ustar,
        lambda _: sample_left_of_contact(),
        lambda _: sample_right_of_contact(),
        None,
    )

def compute_flux(t, gas_state, uL=0.0):
    pstar, _fstar = find_pstar(gas_state)
    rho, u, e = sample_origin(gas_state, pstar, uL)
    p = (GAMMA - 1.0) * rho * e
    E = rho * e + 0.5 * rho * u**2
    flux = jnp.array([
        rho * u,
        rho * u**2 + p,
        u * (E + p),
    ])
    return flux


def compute_integrated_flux(t, gas_state, uL=0.0):
    flux = compute_flux(t, gas_state, uL)
    return t * flux


if __name__ == "__main__":
    gas_state = jnp.array([0.1, 0.1, 0.1])
    pstar, f_star = find_pstar(gas_state)
    print('pstar:', pstar)
    print('f_star:', f_star)
    print()

    rho, u, e = sample_origin(gas_state, pstar)
    print('rho:', rho)
    print('u:', u)
    print('e:', e)
    print()

    flux = compute_flux(1.0, gas_state, pstar)
    print('mass flux:', flux[0])
    print('momentum flux:', flux[1])
    print('energy flux:', flux[2])

    print()
    print("=== Known analytic solutions ===")

    def check(label, gas_state, expected_pstar):
        pstar, residual = find_pstar(gas_state)
        err = float(jnp.abs(pstar - expected_pstar))
        print(f"{label}:")
        print(f"  gas_state = {gas_state}")
        print(f"  pstar     = {pstar}  (expected {expected_pstar})")
        print(f"  |err|     = {err:.2e}, residual = {float(residual):.2e}")

    # 1. Trivial constant state: identical L/R, no waves => p* = 1.
    check(
        "constant state",
        jnp.array([0.0, 0.0, 0.0]),
        1.0,
    )

    # 2. Stationary contact discontinuity: pL = pR, uL = uR, only a density
    #    jump. The Riemann fan is a single stationary contact => p* = 1.
    check(
        "stationary contact",
        jnp.array([0.3, 0.0, 0.0]),
        1.0,
    )

    # 3. Symmetric two-rarefaction (drho = dp = 0, du > 0). The jump function
    #    collapses to (p^ALPHA - 1) + du = 0, giving p* = (1 - du)^(1/ALPHA).
    du = 0.2
    check(
        "symmetric two-rarefaction",
        jnp.array([0.0, 0.0, du]),
        (1.0 - du) ** (1.0 / ALPHA),
    )
