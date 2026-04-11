"""
1D Euler equations solver comparing Exact, Roe, and HLL Riemann solvers.
Uses JAX for vectorised array operations (Roe, HLL) and NumPy for the
scalar Newton‑iteration exact solver.
"""

import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── physical / derived constants (ideal gas, γ = 1.4) ────────────────────
GAMMA = 1.4
GM1   = GAMMA - 1.0          # 0.4
GP1   = GAMMA + 1.0          # 2.4
G1    = GM1 / (2*GAMMA)      # (γ‑1)/(2γ)
G2    = GP1 / (2*GAMMA)      # (γ+1)/(2γ)
G3    = GM1 / GP1             # (γ‑1)/(γ+1)
G4    = 2.0 / GP1             # 2/(γ+1)
G5    = 2.0 / GM1             # 2/(γ‑1)

FLOOR = 1e-12

# ══════════════════════════════════════════════════════════════════════════
#  Conversions  (JAX arrays, shape [3, N])
# ══════════════════════════════════════════════════════════════════════════
def prim2cons(rho, u, p):
    return jnp.stack([rho, rho*u, p/GM1 + 0.5*rho*u**2])

def cons2prim(U):
    rho = U[0]
    u   = U[1] / rho
    p   = GM1 * (U[2] - 0.5*rho*u**2)
    return rho, u, p

def flux_prim(rho, u, p):
    E = p/GM1 + 0.5*rho*u**2
    return jnp.stack([rho*u, rho*u**2 + p, u*(E + p)])

# ══════════════════════════════════════════════════════════════════════════
#  Exact Riemann solver  (scalar / NumPy – called per interface)
# ══════════════════════════════════════════════════════════════════════════
def _fK(p, rhoK, pK, cK):
    """Pressure function f_K and its derivative for one side."""
    if p > pK:                              # shock
        AK = 2.0 / (GP1 * rhoK)
        BK = G3 * pK
        q  = np.sqrt(AK / (p + BK))
        f  = (p - pK) * q
        fp = q * (1.0 - (p - pK) / (2.0*(p + BK)))
    else:                                   # rarefaction
        ratio = max(p / pK, 1e-30)
        f  = G5 * cK * (ratio**G1 - 1.0)
        fp = (1.0/(rhoK*cK)) * ratio**(-G2)
    return f, fp

def solve_star(rhoL, uL, pL, rhoR, uR, pR):
    """Newton iteration for (p*, u*).  Returns (pstar, ustar, is_vacuum)."""
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)
    du = uR - uL

    if du >= G5*(cL + cR):                  # vacuum generated
        return -1.0, 0.5*(uL + uR), True

    # two‑rarefaction initial guess (Toro §4.3.2)
    denom = cL/pL**G1 + cR/pR**G1
    p0 = max(1e-15, ((cL + cR - 0.5*GM1*du) / denom)**(1.0/G1))

    for _ in range(80):
        fL, fpL = _fK(p0, rhoL, pL, cL)
        fR, fpR = _fK(p0, rhoR, pR, cR)
        res = fL + fR + du
        jac = fpL + fpR
        if abs(jac) < 1e-40:
            break
        dp = -res / jac
        p1 = p0 + dp
        if p1 < 0:
            p1 = p0 * 0.5                  # bisection safeguard
        if abs(dp) / (1.0 + 0.5*(p0 + p1)) < 1e-13:
            p0 = p1
            break
        p0 = p1

    pstar = p0
    fL, _ = _fK(pstar, rhoL, pL, cL)
    fR, _ = _fK(pstar, rhoR, pR, cR)
    ustar = 0.5*(uL + uR + fR - fL)
    return pstar, ustar, False

def _sample(xi, rhoL, uL, pL, rhoR, uR, pR, ps, us, vac):
    """Sample exact RP at x/t = xi.  Returns (rho, u, p)."""
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)

    # When p* is negligible the star region is vanishingly thin; use the
    # vacuum sampling to avoid sensitivity from the disappearing star slab.
    if not vac and ps < 1e-6 * max(pL, pR):
        vac = True

    if vac:
        stL = uL + G5*cL
        stR = uR - G5*cR
        if xi <= uL - cL:
            return rhoL, uL, pL
        elif xi <= stL:
            c = G4*(cL + 0.5*GM1*(uL - xi))
            u = G4*(cL + 0.5*GM1*uL + xi)
            r = rhoL*(c/cL)**G5
            p = pL*(c/cL)**(G5*GAMMA)
            return max(r, FLOOR), u, max(p, FLOOR)
        elif xi <= stR:
            return FLOOR, 0.5*(uL + uR), FLOOR
        elif xi <= uR + cR:
            c = G4*(cR - 0.5*GM1*(uR - xi))
            u = G4*(-cR + 0.5*GM1*uR + xi)
            r = rhoR*(c/cR)**G5
            p = pR*(c/cR)**(G5*GAMMA)
            return max(r, FLOOR), u, max(p, FLOOR)
        else:
            return rhoR, uR, pR

    if xi <= us:  # left of contact
        if ps >= pL:                        # left shock
            SL = uL - cL*np.sqrt(G2*ps/pL + G1)
            if xi <= SL:
                return rhoL, uL, pL
            rhoS = rhoL*(ps/pL + G3)/(G3*ps/pL + 1.0)
            return rhoS, us, ps
        else:                               # left rarefaction
            csL = cL*(ps/pL)**G1
            if xi <= uL - cL:
                return rhoL, uL, pL
            elif xi <= us - csL:
                c = G4*(cL + 0.5*GM1*(uL - xi))
                u = G4*(cL + 0.5*GM1*uL + xi)
                r = rhoL*(c/cL)**G5
                p = pL*(c/cL)**(G5*GAMMA)
                return r, u, p
            else:
                return rhoL*(ps/pL)**(1.0/GAMMA), us, ps
    else:          # right of contact
        if ps >= pR:                        # right shock
            SR = uR + cR*np.sqrt(G2*ps/pR + G1)
            if xi >= SR:
                return rhoR, uR, pR
            rhoS = rhoR*(ps/pR + G3)/(G3*ps/pR + 1.0)
            return rhoS, us, ps
        else:                               # right rarefaction
            csR = cR*(ps/pR)**G1
            if xi >= uR + cR:
                return rhoR, uR, pR
            elif xi >= us + csR:
                c = G4*(cR - 0.5*GM1*(uR - xi))
                u = G4*(-cR + 0.5*GM1*uR + xi)
                r = rhoR*(c/cR)**G5
                p = pR*(c/cR)**(G5*GAMMA)
                return r, u, p
            else:
                return rhoR*(ps/pR)**(1.0/GAMMA), us, ps

def exact_godunov_flux(rhoL, uL, pL, rhoR, uR, pR):
    ps, us, vac = solve_star(rhoL, uL, pL, rhoR, uR, pR)
    r, u, p = _sample(0.0, rhoL, uL, pL, rhoR, uR, pR, ps, us, vac)
    E = p/GM1 + 0.5*r*u**2
    return np.array([r*u, r*u**2 + p, u*(E + p)])

# ══════════════════════════════════════════════════════════════════════════
#  Roe solver  (vectorised, JAX)
# ══════════════════════════════════════════════════════════════════════════
def roe_fluxes(rhoL, uL, pL, rhoR, uR, pR):
    EL = pL/GM1 + 0.5*rhoL*uL**2
    ER = pR/GM1 + 0.5*rhoR*uR**2
    HL = (EL + pL)/jnp.maximum(rhoL, FLOOR)
    HR = (ER + pR)/jnp.maximum(rhoR, FLOOR)

    sL = jnp.sqrt(jnp.maximum(rhoL, FLOOR))
    sR = jnp.sqrt(jnp.maximum(rhoR, FLOOR))
    d  = sL + sR

    u_h = (sL*uL + sR*uR) / d
    H_h = (sL*HL + sR*HR) / d
    c2  = GM1*(H_h - 0.5*u_h**2)
    c_h = jnp.sqrt(jnp.maximum(c2, FLOOR))
    rho_h = sL * sR

    l1 = u_h - c_h;  l2 = u_h;  l3 = u_h + c_h

    # Harten entropy fix
    eps = 0.15 * c_h
    al1 = jnp.where(jnp.abs(l1) < eps, (l1**2 + eps**2)/(2*eps), jnp.abs(l1))
    al2 = jnp.abs(l2)
    al3 = jnp.where(jnp.abs(l3) < eps, (l3**2 + eps**2)/(2*eps), jnp.abs(l3))

    drho = rhoR - rhoL;  du = uR - uL;  dp = pR - pL
    a1 = (dp - rho_h*c_h*du) / (2*c_h**2)
    a2 = drho - dp / c_h**2
    a3 = (dp + rho_h*c_h*du) / (2*c_h**2)

    d0 = al1*a1 + al2*a2 + al3*a3
    d1 = al1*a1*(u_h - c_h) + al2*a2*u_h + al3*a3*(u_h + c_h)
    d2 = (al1*a1*(H_h - u_h*c_h) + al2*a2*0.5*u_h**2
          + al3*a3*(H_h + u_h*c_h))

    FL = flux_prim(rhoL, uL, pL)
    FR = flux_prim(rhoR, uR, pR)
    return 0.5*(FL + FR) - 0.5*jnp.stack([d0, d1, d2])

# ══════════════════════════════════════════════════════════════════════════
#  HLL solver  (vectorised, JAX)
# ══════════════════════════════════════════════════════════════════════════
def hll_fluxes(rhoL, uL, pL, rhoR, uR, pR):
    cL = jnp.sqrt(GAMMA * jnp.maximum(pL, FLOOR) / jnp.maximum(rhoL, FLOOR))
    cR = jnp.sqrt(GAMMA * jnp.maximum(pR, FLOOR) / jnp.maximum(rhoR, FLOOR))

    # Roe‑averaged wave speeds (Einfeldt / HLLE)
    sL = jnp.sqrt(jnp.maximum(rhoL, FLOOR))
    sR = jnp.sqrt(jnp.maximum(rhoR, FLOOR))
    u_h = (sL*uL + sR*uR) / (sL + sR)
    EL = pL/GM1 + 0.5*rhoL*uL**2
    ER = pR/GM1 + 0.5*rhoR*uR**2
    H_h = (sL*(EL+pL)/jnp.maximum(rhoL,FLOOR)
         + sR*(ER+pR)/jnp.maximum(rhoR,FLOOR)) / (sL + sR)
    c_h = jnp.sqrt(jnp.maximum(GM1*(H_h - 0.5*u_h**2), FLOOR))

    SL = jnp.minimum(uL - cL, u_h - c_h)
    SR = jnp.maximum(uR + cR, u_h + c_h)

    FL = flux_prim(rhoL, uL, pL)
    FR = flux_prim(rhoR, uR, pR)
    UL = jnp.stack([rhoL, rhoL*uL, EL])
    UR = jnp.stack([rhoR, rhoR*uR, ER])

    dS = jnp.where(jnp.abs(SR - SL) < 1e-30, 1e-30, SR - SL)
    F_hll = (SR*FL - SL*FR + SL*SR*(UR - UL)) / dS

    return jnp.where(SL >= 0, FL, jnp.where(SR <= 0, FR, F_hll))

# ══════════════════════════════════════════════════════════════════════════
#  MUSCL reconstruction helpers
# ══════════════════════════════════════════════════════════════════════════
def _slope_limit(dL, dR, limiter="mc"):
    """Apply a slope limiter to left/right differences."""
    if limiter == "minmod":
        return jnp.where(dL * dR <= 0, 0.0,
                         jnp.where(jnp.abs(dL) < jnp.abs(dR), dL, dR))
    elif limiter == "vanleer":
        s = dL * dR
        return jnp.where(s <= 0, 0.0,
                         2.0 * s / (dL + dR + jnp.where(s <= 0, 1.0, 0.0)))
    else:  # mc (monotonized central)
        c = 0.5 * (dL + dR)
        s = jnp.sign(c)
        return s * jnp.maximum(0.0, jnp.minimum(
            jnp.abs(c), jnp.minimum(2.0*jnp.abs(dL), 2.0*jnp.abs(dR))))

def _muscl_reconstruct(rho, u, p, limiter="mc"):
    """Linear reconstruction from extended primitives (N+4,) to interface
    left/right states (N+1,) each."""
    # fall back to first-order in near-vacuum cells to avoid extreme sound speeds
    rho_c = rho[1:-1]
    # use minimum density of cell and its neighbours for robustness
    rho_nbr = jnp.minimum(rho[:-2], rho[2:])
    theta = jnp.minimum(1.0, jnp.minimum(rho_c, rho_nbr) / 1e-2)

    def _recon(W):
        dL = W[1:-1] - W[:-2]
        dR = W[2:]   - W[1:-1]
        s  = _slope_limit(dL, dR, limiter) * theta
        W_L = W[1:-1] - 0.5 * s   # left  face of each cell
        W_R = W[1:-1] + 0.5 * s   # right face of each cell
        return W_L, W_R

    rL, rR = _recon(rho)
    uL, uR = _recon(u)
    pL, pR = _recon(p)

    # floor density and pressure to prevent negatives near strong gradients
    rR = jnp.maximum(rR, FLOOR);  rL = jnp.maximum(rL, FLOOR)
    pR = jnp.maximum(pR, FLOOR);  pL = jnp.maximum(pL, FLOOR)

    # interface i: left state = right-face of cell i, right state = left-face of cell i+1
    return rR[:-1], uR[:-1], pR[:-1], rL[1:], uL[1:], pL[1:]

# ══════════════════════════════════════════════════════════════════════════
#  Godunov finite‑volume time stepper
# ══════════════════════════════════════════════════════════════════════════
def _extend(rho, u, p, bc):
    """Add one ghost cell on each side."""
    if bc == "reflecting":
        return (jnp.concatenate([rho[:1], rho, rho[-1:]]),
                jnp.concatenate([-u[:1], u, -u[-1:]]),
                jnp.concatenate([p[:1],   p,  p[-1:]]))
    # transmissive
    return (jnp.concatenate([rho[:1], rho, rho[-1:]]),
            jnp.concatenate([u[:1],   u,   u[-1:]]),
            jnp.concatenate([p[:1],   p,   p[-1:]]))

def _extend2(rho, u, p, bc):
    """Add two ghost cells on each side (for MUSCL reconstruction)."""
    if bc == "reflecting":
        return (jnp.concatenate([rho[1:2], rho[:1], rho, rho[-1:], rho[-2:-1]]),
                jnp.concatenate([-u[1:2], -u[:1],   u,  -u[-1:], -u[-2:-1]]),
                jnp.concatenate([p[1:2],   p[:1],   p,   p[-1:],  p[-2:-1]]))
    # transmissive
    return (jnp.concatenate([rho[:1], rho[:1], rho, rho[-1:], rho[-1:]]),
            jnp.concatenate([u[:1],   u[:1],   u,   u[-1:],  u[-1:]]),
            jnp.concatenate([p[:1],   p[:1],   p,   p[-1:],  p[-1:]]))

def solve(rho0, u0, p0, dx, t_end, solver="exact", bc="transmissive",
          cfl=0.45, order=1, limiter="mc"):
    N = len(rho0)
    U = prim2cons(jnp.asarray(rho0, dtype=float),
                  jnp.asarray(u0,   dtype=float),
                  jnp.asarray(p0,   dtype=float))
    t = 0.0
    step = 0
    while t < t_end and step < 200000:
        rho, u, p = cons2prim(U)
        rho = jnp.maximum(rho, FLOOR)
        # When pressure goes negative, kinetic energy has exceeded total
        # energy — cap velocity to restore energy consistency.
        neg_p = p < 0.0
        u_max = jnp.sqrt(jnp.maximum(2.0 * U[2] / rho, 0.0))
        u = jnp.where(neg_p, jnp.clip(u, -u_max, u_max), u)
        p   = jnp.maximum(p,   FLOOR)
        U   = prim2cons(rho, u, p)

        c   = jnp.sqrt(GAMMA * p / rho)
        smax = float(jnp.max(jnp.abs(u) + c))
        dt  = min(cfl * dx / smax, t_end - t)

        if order == 1:
            rE, uE, pE = _extend(rho, u, p, bc)
            rL_f, uL_f, pL_f = rE[:-1], uE[:-1], pE[:-1]
            rR_f, uR_f, pR_f = rE[1:],  uE[1:],  pE[1:]
        else:
            rE, uE, pE = _extend2(rho, u, p, bc)
            rL_f, uL_f, pL_f, rR_f, uR_f, pR_f = _muscl_reconstruct(
                rE, uE, pE, limiter)

        if solver == "exact":
            F = np.zeros((3, N+1))
            rL_n = np.asarray(rL_f); uL_n = np.asarray(uL_f); pL_n = np.asarray(pL_f)
            rR_n = np.asarray(rR_f); uR_n = np.asarray(uR_f); pR_n = np.asarray(pR_f)
            for i in range(N+1):
                F[:, i] = exact_godunov_flux(rL_n[i], uL_n[i], pL_n[i],
                                             rR_n[i], uR_n[i], pR_n[i])
            F = jnp.asarray(F)
        elif solver == "roe":
            F = roe_fluxes(rL_f, uL_f, pL_f, rR_f, uR_f, pR_f)
        else:  # hll
            F = hll_fluxes(rL_f, uL_f, pL_f, rR_f, uR_f, pR_f)

        U = U - (dt/dx) * (F[:, 1:] - F[:, :-1])
        t += dt
        step += 1

    rho, u, p = cons2prim(U)
    return (np.asarray(jnp.maximum(rho, FLOOR)),
            np.asarray(u),
            np.asarray(jnp.maximum(p, FLOOR)))

# ══════════════════════════════════════════════════════════════════════════
#  Exact (analytical) reference for a single Riemann problem
# ══════════════════════════════════════════════════════════════════════════
def exact_ref(x, x0, t, wL, wR, npts=2000):
    """wL = (rhoL, uL, pL) etc.  Returns arrays (rho, u, p) on a fine grid
    *and* on the supplied cell‑centre grid x."""
    rhoL, uL, pL = wL
    rhoR, uR, pR = wR
    ps, us, vac = solve_star(rhoL, uL, pL, rhoR, uR, pR)

    xf = np.linspace(x[0], x[-1], npts)
    def _s(xi):
        return _sample((xi - x0)/t if t > 0 else 0.0,
                        rhoL, uL, pL, rhoR, uR, pR, ps, us, vac)
    rf, uf, pf = np.array([_s(xi) for xi in xf]).T
    return xf, rf, uf, pf

# ══════════════════════════════════════════════════════════════════════════
#  Test‑case definitions
# ══════════════════════════════════════════════════════════════════════════
TESTS = {
    "1_sod": dict(
        title  = "Test 1 – Sod shock tube (baseline)",
        wL     = (1.0,   0.0, 1.0),
        wR     = (0.125, 0.0, 0.1),
        x0     = 0.5,
        domain = (0.0, 1.0),
        t_end  = 0.2,
        N      = 200,
        bc     = "transmissive",
    ),
    "2_einfeldt": dict(
        title  = "Test 2 – Einfeldt 1‑2‑3  (near‑vacuum / entropy fix)",
        wL     = (1.0, -2.0, 0.4),
        wR     = (1.0,  2.0, 0.4),
        x0     = 0.5,
        domain = (0.0, 1.0),
        t_end  = 0.15,
        N      = 200,
        bc     = "transmissive",
    ),
    "3_vacuum": dict(
        title  = "Test 3 – vacuum generation",
        wL     = (1.0, -4.0, 0.4),
        wR     = (1.0,  4.0, 0.4),
        x0     = 0.5,
        domain = (0.0, 1.0),
        t_end  = 0.15,
        N      = 200,
        bc     = "transmissive",
    ),
    "4_strong_shock": dict(
        title  = "Test 4 – strong shock  (pL/pR = 10⁷)",
        wL     = (1.0, 0.0, 1e5),
        wR     = (1.0, 0.0, 0.01),
        x0     = 0.5,
        domain = (0.0, 1.0),
        t_end  = 0.001,
        N      = 200,
        bc     = "transmissive",
    ),
    "5_contact": dict(
        title  = "Test 5 – isolated contact discontinuity",
        wL     = (1.0,   0.0, 1.0),
        wR     = (0.125, 0.0, 1.0),
        x0     = 0.5,
        domain = (0.0, 1.0),
        t_end  = 0.3,
        N      = 200,
        bc     = "transmissive",
    ),
}

# ── Woodward‑Colella is special (two discontinuities, reflecting BCs) ───
WC = dict(
    title  = "Test 6 – Woodward–Colella blast wave",
    domain = (0.0, 1.0),
    t_end  = 0.038,
    N      = 400,
    bc     = "reflecting",
)

# ══════════════════════════════════════════════════════════════════════════
#  Initial conditions
# ══════════════════════════════════════════════════════════════════════════
def make_ic_riemann(N, domain, x0, wL, wR):
    xL, xR = domain
    dx = (xR - xL) / N
    x  = np.linspace(xL + 0.5*dx, xR - 0.5*dx, N)
    rho = np.where(x < x0, wL[0], wR[0])
    u   = np.where(x < x0, wL[1], wR[1])
    p   = np.where(x < x0, wL[2], wR[2])
    return x, dx, rho, u, p

def make_ic_wc(N, domain):
    xL, xR = domain
    dx = (xR - xL) / N
    x  = np.linspace(xL + 0.5*dx, xR - 0.5*dx, N)
    rho = np.ones(N)
    u   = np.zeros(N)
    p   = np.where(x < 0.1, 1000.0, np.where(x > 0.9, 100.0, 0.01))
    return x, dx, rho, u, p

# ══════════════════════════════════════════════════════════════════════════
#  Plotting helper
# ══════════════════════════════════════════════════════════════════════════
SOLVER_STYLE = dict(
    exact      = dict(color="tab:blue",   ls="-",  lw=1.2, label="Exact (Godunov)"),
    roe        = dict(color="tab:red",    ls="--", lw=1.2, label="Roe"),
    hll        = dict(color="tab:green",  ls="-.", lw=1.2, label="HLL"),
    exact_muscl= dict(color="tab:cyan",   ls="-",  lw=1.2, label="Exact+MUSCL"),
    roe_muscl  = dict(color="tab:orange", ls="--", lw=1.2, label="Roe+MUSCL"),
    hll_muscl  = dict(color="tab:purple", ls="-.", lw=1.2, label="HLL+MUSCL"),
)

def plot_case(name, x, results, ref=None, title=""):
    """results = {solver_name: (rho, u, p)}, ref = (xf, rf, uf, pf) or None."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = [r"$\rho$", r"$u$", r"$p$"]

    for ax, lab, idx in zip(axes, labels, range(3)):
        if ref is not None:
            xf = ref[0]
            ax.plot(xf, ref[idx+1], "k-", lw=1.8, alpha=0.45, label="Analytic")
        for s in SOLVER_STYLE:
            if s in results:
                ax.plot(x, results[s][idx], **SOLVER_STYLE[s])
        ax.set_xlabel("x")
        ax.set_ylabel(lab)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join("./outputs", f"{name}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out

# ══════════════════════════════════════════════════════════════════════════
#  Run everything
# ══════════════════════════════════════════════════════════════════════════
def run_single_rp(name, cfg):
    print(f"\n{'='*60}")
    print(f"  {cfg['title']}")
    print(f"{'='*60}")
    x, dx, rho0, u0, p0 = make_ic_riemann(
        cfg["N"], cfg["domain"], cfg["x0"], cfg["wL"], cfg["wR"])

    results = {}
    for s in ("exact", "roe", "hll"):
        print(f"  running {s:6s} ... ", end="", flush=True)
        try:
            r, u, p = solve(rho0, u0, p0, dx, cfg["t_end"],
                            solver=s, bc=cfg["bc"])
            results[s] = (r, u, p)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")
    for s in ("exact", "roe", "hll"):
        key = f"{s}_muscl"
        print(f"  running {key:12s} ... ", end="", flush=True)
        try:
            r, u, p = solve(rho0, u0, p0, dx, cfg["t_end"],
                            solver=s, bc=cfg["bc"], order=2)
            results[key] = (r, u, p)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")

    # analytical reference
    ref = exact_ref(x, cfg["x0"], cfg["t_end"], cfg["wL"], cfg["wR"])
    return plot_case(name, x, results, ref=ref, title=cfg["title"])

def run_wc():
    cfg = WC
    print(f"\n{'='*60}")
    print(f"  {cfg['title']}")
    print(f"{'='*60}")
    x, dx, rho0, u0, p0 = make_ic_wc(cfg["N"], cfg["domain"])

    results = {}
    for s in ("exact", "roe", "hll"):
        print(f"  running {s:6s} (N={cfg['N']}) ... ", end="", flush=True)
        try:
            r, u, p = solve(rho0, u0, p0, dx, cfg["t_end"],
                            solver=s, bc=cfg["bc"])
            results[s] = (r, u, p)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")
    for s in ("exact", "roe", "hll"):
        key = f"{s}_muscl"
        print(f"  running {key:12s} (N={cfg['N']}) ... ", end="", flush=True)
        try:
            r, u, p = solve(rho0, u0, p0, dx, cfg["t_end"],
                            solver=s, bc=cfg["bc"], order=2)
            results[key] = (r, u, p)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")

    # no simple analytical reference for WC – use exact solver as reference
    return plot_case("6_woodward_colella", x, results, ref=None,
                     title=cfg["title"])

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)
    output_files = []

    for name, cfg in TESTS.items():
        f = run_single_rp(name, cfg)
        output_files.append(f)

    f = run_wc()
    output_files.append(f)

    print(f"\n✓ All done – {len(output_files)} plots saved.")