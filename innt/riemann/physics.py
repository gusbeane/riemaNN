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
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from typing import NamedTuple

GAMMA: float = 5.0 / 3.0
ALPHA: float = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA: float = (GAMMA - 1.0) / (GAMMA + 1.0)
MU: float = (GAMMA - 1.0) / 2.0

GAS_STATE_DIM: int = 6

def flux_from_primitive_state(rho: float, u: float, p: float):
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return jnp.array([
        rho * u,
        rho * u**2 + p,
        u * (E + p),
    ])

class GasState(NamedTuple):
    log10_rhoL: jax.Array
    uL:         jax.Array
    log10_pL:   jax.Array
    log10_rhoR: jax.Array
    log10_pR:   jax.Array
    uR:        jax.Array

    @classmethod
    def from_array(cls, x):
        return cls(
            log10_rhoL=x[..., 0],
            uL=x[...,1],
            log10_pL=x[..., 2],
            log10_rhoR=x[..., 3],
            uR=x[..., 4],
            log10_pR=x[..., 5],
        )
    
    @classmethod
    def from_linear(cls, *, rhoL: float, uL: float, pL: float, rhoR: float, uR: float, pR: float):
        return cls(
            log10_rhoL=jnp.log10(rhoL),
            uL=uL,
            log10_pL=jnp.log10(pL),
            log10_rhoR=jnp.log10(rhoR),
            uR=uR,
            log10_pR=jnp.log10(pR),
        )
   
    def as_array(self):
        return jnp.stack([self.log10_rhoL, self.uL, self.log10_pL, self.log10_rhoR, self.uR, self.log10_pR], axis=-1)
    
    @property
    def rhoL(self):
        return 10.0 ** self.log10_rhoL
    
    @property
    def pL(self):
        return 10.0 ** self.log10_pL
    
    @property
    def rhoR(self):
        return 10.0 ** self.log10_rhoR
    
    @property
    def pR(self):
        return 10.0 ** self.log10_pR

    @property
    def aL(self):
        return jnp.sqrt(GAMMA * self.pL / self.rhoL)
    
    @property
    def aR(self):
        return jnp.sqrt(GAMMA * self.pR / self.rhoR)

    @property
    def ucrit(self):
        return (self.aL + self.aR) / MU
    
    @property
    def uRL(self):
        return self.uR - self.uL
    
    @property
    def is_vacuum(self):
        return self.uRL >= self.ucrit
    
    def rhoK(self, LR: int):
        return jnp.where(LR == -1, self.rhoL, self.rhoR)

    def pK(self, LR: int):
        return jnp.where(LR == -1, self.pL, self.pR)
    
    def aK(self, LR: int):
        return jnp.where(LR == -1, self.aL, self.aR)

@jax.jit
def abs_flux_jacobian_from_primitive_state(rho: float, u: float, p: float):
    """Return |F'(U)| for the 1D Euler equations.

    Conservative variable convention:
        U = [rho, rho*u, E]

    This computes
        |A| = R @ abs(Lambda) @ inv(R)

    where A = F'(U).
    """

    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    H = (E + p) / rho
    a = jnp.sqrt(GAMMA * p / rho)

    R = jnp.array([
        [1.0,       1.0, 1.0],
        [u - a,     u,   u + a],
        [H - u*a, 0.5 * u * u, H + u*a],
    ])

    lam = jnp.array([u - a, u, u + a])

    return R @ jnp.diag(jnp.abs(lam)) @ jnp.linalg.inv(R)

@jax.jit
def abs_flux_jacobian_from_conserved_state(U):
    """Return |F'(U)| from U = [rho, rho*u, E]."""

    rho = U[0]
    mom = U[1]
    E = U[2]

    u = mom / rho
    p = (GAMMA - 1.0) * (E - 0.5 * mom * u)

    return abs_flux_jacobian_from_primitive_state(rho, u, p)

@jax.jit
def ftilde_one(p: float, gs: GasState, LR: int):
    """Contribution of one side (K = L or R) to the Riemann star-pressure equation.

    LR = -1 for L, +1 for R.
    """

    rhoK, pK, aK = gs.rhoK(LR), gs.pK(LR), gs.aK(LR)
    AK = 2/((GAMMA + 1.) * rhoK)
    BK = BETA * pK

    shock = (p - pK) * jnp.sqrt(AK / (p + BK))
    rarefaction = (aK / MU) * ((p / pK) ** ALPHA - 1.)

    return jnp.where(p > pK, shock, rarefaction)


@jax.jit
def fstar_one(p: float, gs: GasState):
    """Residual of the Riemann star-pressure equation; zero at the true p*."""
    return ftilde_one(p, gs, -1) + ftilde_one(p, gs, +1) + gs.uRL


dfstar_dp_one = jax.grad(fstar_one, argnums=0)


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
def _newton(gs: GasState):
    """Newton iteration for p*. Fast but can diverge."""

    p0 = jnp.maximum(two_rarefaction_p0(gs), 1e-30)

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
        fstar_ = fstar_one(pstar, gs)
        dfstar_ = dfstar_dp_one(pstar, gs)
        pstar = pstar - fstar_ / dfstar_
        return pstar, pstar_prev, fstar_one(pstar, gs), i + 1

    init = (p0, jnp.inf, fstar_one(p0, gs), 0)
    pstar, _pstar_prev, fstar_final, _ = jax.lax.while_loop(cond, body, init)
    return pstar, fstar_final


@jax.jit
def _bisect(gs: GasState):
    """Bisection solver for p*. Slow but guaranteed to converge.

    Vacuum states (no physical root) return a very small p* with a
    non-zero residual.
    """
    p_guess = jnp.maximum(two_rarefaction_p0(gs), 1e-30)

    p_lo = p_guess * 0.5
    p_hi = p_guess * 2.0
    f_lo = fstar_one(p_lo, gs)
    f_hi = fstar_one(p_hi, gs)

    def widen_cond(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        return (_f_lo * _f_hi > 0) & (i < 60)

    def widen_body(state):
        _p_lo, _p_hi, _f_lo, _f_hi, i = state
        _p_lo = jnp.where(_f_lo > 0, jnp.maximum(_p_lo * 0.01, 1e-30), _p_lo)
        _p_hi = jnp.where(_f_hi < 0, _p_hi * 100.0, _p_hi)
        return (
            _p_lo, _p_hi,
            fstar_one(_p_lo, gs), fstar_one(_p_hi, gs), i + 1,
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
        f_mid = fstar_one(p_mid, gs)
        _p_lo = jnp.where(f_mid < 0, p_mid, _p_lo)
        _p_hi = jnp.where(f_mid >= 0, p_mid, _p_hi)
        return _p_lo, _p_hi, i + 1

    p_lo, p_hi, _ = jax.lax.while_loop(
        bisect_cond, bisect_body, (p_lo, p_hi, 0)
    )

    p_result = 0.5 * (p_lo + p_hi)
    f_result = fstar_one(p_result, gs)
    return p_result, f_result


@jax.jit
def find_pstar(gs: GasState):
    """Find p* via Newton with bisection fallback; returns (pstar, residual).
    
    Returns (p*=0, 0) if uRL > critical velocity (vacuum state).
    """
    
    def normal_case(_):
        p_newton, f_newton = _newton(gs)
        newton_ok = (
            jnp.isfinite(p_newton)
            & jnp.isfinite(f_newton)
            & (p_newton > 0)
            & (jnp.abs(f_newton) < 1e-12)
        )
        p_bisect, f_bisect = jax.lax.cond(
            newton_ok,
            lambda _: (p_newton, f_newton),
            lambda _: _bisect(gs),
            None,
        )
        return p_bisect, f_bisect

    pstar, fresid = jax.lax.cond(
        gs.is_vacuum,
        lambda _: (0.0, 0.0),
        normal_case,
        None,
    )
    return pstar, fresid


@jax.jit
def sample_origin(gs: GasState, pstar: float):
    """Return rho, u, e at x=0, t>0.

    Velocities are in units of u_ref = sqrt(p_ref / rho_ref).
    Pressure and density use p_ref = rho_ref = 1 normalization.

    uL fixes the Galilean frame. If uL=0, then uR = du * ducrit.
    """

    rhoL, uL, pL, rhoR, uR, pR= gs.rhoL, gs.uL, gs.pL, gs.rhoR, gs.uR, gs.pR

    aL, aR = gs.aL, gs.aR

    fL = ftilde_one(pstar, gs, -1)
    fR = ftilde_one(pstar, gs, +1)

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

    def sample_vacuum():
        # Two rarefactions with a vacuum band in between; no contact / star
        # states. Wave structure at xi: L | left fan | vacuum | right fan | R.
        SHL = uL - aL                 # left rarefaction head
        S_star_L = uL + aL / MU       # left vacuum front (fan tail)
        S_star_R = uR - aR / MU       # right vacuum front (fan tail)
        SHR = uR + aR                 # right rarefaction head

        zero = jnp.zeros_like(rhoL)
        vacuum_state = (zero, zero, zero)  # rho = p = 0 -> flux = 0

        return jax.lax.cond(
            xi <= SHL,
            lambda _: (rhoL, uL, eL),
            lambda _: jax.lax.cond(
                xi <= S_star_L,
                lambda __: left_fan_state(),
                lambda __: jax.lax.cond(
                    xi < S_star_R,
                    lambda ___: vacuum_state,
                    lambda ___: jax.lax.cond(
                        xi <= SHR,
                        lambda ____: right_fan_state(),
                        lambda ____: (rhoR, uR, eR),
                        None,
                    ),
                    None,
                ),
                None,
            ),
            None,
        )

    def sample_normal():
        # Contact speed separates left-star and right-star sides.
        return jax.lax.cond(
            xi <= ustar,
            lambda _: sample_left_of_contact(),
            lambda _: sample_right_of_contact(),
            None,
        )

    return jax.lax.cond(
        gs.is_vacuum,
        lambda _: sample_vacuum(),
        lambda _: sample_normal(),
        None,
    )

def compute_flux(t, gas_state):
    pstar, _fstar = find_pstar(gas_state)
    rho, u, e = sample_origin(gas_state, pstar)
    p = (GAMMA - 1.0) * rho * e
    E = rho * e + 0.5 * rho * u**2
    flux = jnp.array([
        rho * u,
        rho * u**2 + p,
        u * (E + p),
    ])
    return flux


def compute_integrated_flux(t, gs: GasState):
    flux = compute_flux(t, gs)
    return t * flux


if __name__ == "__main__":
    # gas_state = jnp.array([0.1, 0.1, 0.1])
    gs = GasState.from_linear(rhoL=1.0, uL=-5.0, pL=1.0, rhoR=0.4, uR=10.0, pR=1.0)
    pstar, f_star = find_pstar(gs)
    print('pstar:', pstar)
    print('f_star:', f_star)
    print('is_vacuum:', gs.is_vacuum)
    print()
    
    gs = GasState.from_linear(rhoL=1.0, uL=0.0, pL=1.0, rhoR=0.4, uR=0.3, pR=1.0)
    pstar, f_star = find_pstar(gs)
    print('pstar:', pstar)
    print('f_star:', f_star)
    print('is_vacuum:', gs.is_vacuum)
    print()

    rho, u, e = sample_origin(gs, pstar)
    print('rho:', rho)
    print('u:', u)
    print('e:', e)
    print()

    flux = compute_flux(1.0, gs)
    print('mass flux:', flux[0])
    print('momentum flux:', flux[1])
    print('energy flux:', flux[2])
    print()

    gs = GasState.from_linear(rhoL=1.0, uL=-5., pL=1.0, rhoR=1.0, uR=5., pR=1.0)
    pstar, f_star = find_pstar(gs)
    print('pstar:', pstar)
    print('f_star:', f_star)
    print('u_crit:', gs.ucrit)
    print('is_vacuum:', gs.is_vacuum)
    print()

    rho, u, e = sample_origin(gs, pstar)
    print('rho:', rho)
    print('u:', u)
    print('e:', e)
    print()

    flux = compute_flux(1.0, gs)
    print('mass flux:', flux[0])
    print('momentum flux:', flux[1])
    print('energy flux:', flux[2])