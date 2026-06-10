"""HLLC approximate Riemann solver for 1D Euler (Toro ch. 10), JAX.

Flux at x/t = 0 from primitive left/right states. Wave speeds use Toro's
adaptive noniterative pressure estimate (PVRS / two-rarefaction / two-shock)
with shock-corrected speed bounds (Toro eqs. 10.59-10.60), contact speed from
eq. 10.37, and star states from eq. 10.39.

Run as a script to compare against the exact solver on the training domain:
    JAX_PLATFORMS=cpu python hllc.py [N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import GAMMA, flux_from_primitive_state  # noqa: E402

_Z = (GAMMA - 1.0) / (2.0 * GAMMA)


def _pstar_estimate(rhoL, uL, pL, aL, rhoR, uR, pR, aR):
    """Toro's adaptive noniterative p* estimate (ANRS, sec. 9.5.2)."""
    rho_bar = 0.5 * (rhoL + rhoR)
    a_bar = 0.5 * (aL + aR)
    p_pvrs = 0.5 * (pL + pR) - 0.5 * (uR - uL) * rho_bar * a_bar

    p_min = jnp.minimum(pL, pR)
    p_max = jnp.maximum(pL, pR)

    # two-rarefaction estimate
    num = aL + aR - 0.5 * (GAMMA - 1.0) * (uR - uL)
    den = aL / pL**_Z + aR / pR**_Z
    p_tr = (jnp.maximum(num, 0.0) / den) ** (1.0 / _Z)

    # two-shock estimate, anchored at p0 = max(0, p_pvrs)
    p0 = jnp.maximum(p_pvrs, 0.0)
    AL = 2.0 / ((GAMMA + 1.0) * rhoL)
    AR = 2.0 / ((GAMMA + 1.0) * rhoR)
    BL = (GAMMA - 1.0) / (GAMMA + 1.0) * pL
    BR = (GAMMA - 1.0) / (GAMMA + 1.0) * pR
    gL = jnp.sqrt(AL / (p0 + BL))
    gR = jnp.sqrt(AR / (p0 + BR))
    p_ts = jnp.maximum((gL * pL + gR * pR - (uR - uL)) / (gL + gR), 0.0)

    use_pvrs = (p_max / p_min <= 2.0) & (p_min <= p_pvrs) & (p_pvrs <= p_max)
    return jnp.where(use_pvrs, jnp.maximum(p_pvrs, 0.0), jnp.where(p_pvrs < p_min, p_tr, p_ts))


def hllc_flux(rhoL, uL, pL, rhoR, uR, pR):
    """HLLC flux (mass, momentum, energy) at x/t = 0."""
    aL = jnp.sqrt(GAMMA * pL / rhoL)
    aR = jnp.sqrt(GAMMA * pR / rhoR)
    pstar = _pstar_estimate(rhoL, uL, pL, aL, rhoR, uR, pR, aR)

    # shock-corrected wave speed bounds (Toro 10.59-10.60)
    qL = jnp.where(pstar > pL, jnp.sqrt(1.0 + 0.5 * (GAMMA + 1.0) / GAMMA * (pstar / pL - 1.0)), 1.0)
    qR = jnp.where(pstar > pR, jnp.sqrt(1.0 + 0.5 * (GAMMA + 1.0) / GAMMA * (pstar / pR - 1.0)), 1.0)
    SL = uL - aL * qL
    SR = uR + aR * qR

    # contact speed (Toro 10.37)
    mL = rhoL * (SL - uL)
    mR = rhoR * (SR - uR)
    Sstar = (pR - pL + uL * mL - uR * mR) / (mL - mR)

    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL**2
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR**2
    FL = flux_from_primitive_state(rhoL, uL, pL)
    FR = flux_from_primitive_state(rhoR, uR, pR)
    UL_c = jnp.array([rhoL, rhoL * uL, EL])
    UR_c = jnp.array([rhoR, rhoR * uR, ER])

    def star_flux(rhoK, uK, pK, EK, SK, FK, UK):
        denom = SK - Sstar
        denom = jnp.where(jnp.abs(denom) < 1e-300, jnp.sign(denom) * 1e-300 + 1e-300, denom)
        rho_star = rhoK * (SK - uK) / denom
        e_term = EK / rhoK + (Sstar - uK) * (Sstar + pK / (rhoK * (SK - uK)))
        U_star = rho_star * jnp.array([1.0, Sstar, e_term])
        return FK + SK * (U_star - UK)

    F_starL = star_flux(rhoL, uL, pL, EL, SL, FL, UL_c)
    F_starR = star_flux(rhoR, uR, pR, ER, SR, FR, UR_c)

    return jnp.where(
        SL >= 0.0, FL,
        jnp.where(Sstar >= 0.0, F_starL, jnp.where(SR >= 0.0, F_starR, FR)),
    )


def hllc_flux_x(x):
    """Adapter for the training layout x = (t, log10 rhoL, uL, log10 pL, log10 rhoR, uR, log10 pR)."""
    return hllc_flux(10.0 ** x[1], x[2], 10.0 ** x[3], 10.0 ** x[4], x[5], 10.0 ** x[6])


def roe_flux(rhoL, uL, pL, rhoR, uR, pR):
    """Roe flux (Toro ch. 11), no entropy fix: F = 0.5(FL+FR) - 0.5 sum |lam_k| alpha_k K_k.

    Same structural family as the learned head flux_LR - 0.5 D @ dU with
    D = |A(Roe average)|.
    """
    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL**2
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR**2
    HL = (EL + pL) / rhoL
    HR = (ER + pR) / rhoR

    sL, sR = jnp.sqrt(rhoL), jnp.sqrt(rhoR)
    u_t = (sL * uL + sR * uR) / (sL + sR)
    H_t = (sL * HL + sR * HR) / (sL + sR)
    a2_t = (GAMMA - 1.0) * (H_t - 0.5 * u_t**2)
    a_t = jnp.sqrt(jnp.maximum(a2_t, 1e-300))

    d_rho = rhoR - rhoL
    d_m = rhoR * uR - rhoL * uL
    d_E = ER - EL

    alpha2 = (GAMMA - 1.0) / a2_t * (d_rho * (H_t - u_t**2) + u_t * d_m - d_E)
    alpha1 = (d_rho * (u_t + a_t) - d_m - a_t * alpha2) / (2.0 * a_t)
    alpha3 = d_rho - alpha1 - alpha2

    lam1, lam2, lam3 = jnp.abs(u_t - a_t), jnp.abs(u_t), jnp.abs(u_t + a_t)
    K1 = jnp.array([1.0, u_t - a_t, H_t - u_t * a_t])
    K2 = jnp.array([1.0, u_t, 0.5 * u_t**2])
    K3 = jnp.array([1.0, u_t + a_t, H_t + u_t * a_t])

    FL = flux_from_primitive_state(rhoL, uL, pL)
    FR = flux_from_primitive_state(rhoR, uR, pR)
    diss = lam1 * alpha1 * K1 + lam2 * alpha2 * K2 + lam3 * alpha3 * K3
    return 0.5 * (FL + FR) - 0.5 * diss


def roe_flux_x(x):
    """Adapter for the training layout x = (t, log10 rhoL, uL, log10 pL, log10 rhoR, uR, log10 pR)."""
    return roe_flux(10.0 ** x[1], x[2], 10.0 ** x[3], 10.0 ** x[4], x[5], 10.0 ** x[6])


# ---------------------------------------------------------------------------
# Comparison against the exact solver

def _main(N: int = 500_000) -> None:
    import jax.random as jr
    from physics import GasState, find_pstar, sample_origin
    from train_mat import TRAIN_BOUNDS, F_true, flux_scale, _build_model
    from evaluate import draw_rect

    x = draw_rect(jr.PRNGKey(11), N, TRAIN_BOUNDS)
    ft = jax.vmap(F_true)(x)
    fh = jax.vmap(hllc_flux_x)(x)
    S = flux_scale(x)
    serr = jnp.abs(fh - ft) / S
    loss_pt = (serr**2).sum(axis=1)
    names = ["mass", "momentum", "energy"]

    print(f"HLLC vs exact on {N} samples, training metric |dF|/S:")
    print(f"  overall loss mean((dF/S)^2): {float(jnp.mean(serr**2)):.4e}")
    for i, ch in enumerate(names):
        q = jnp.percentile(serr[:, i], jnp.array([50.0, 99.0, 99.9, 100.0]))
        n01 = max(1, N // 1000)
        share = float(jnp.sort(serr[:, i] ** 2)[-n01:].sum() / (serr[:, i] ** 2).sum())
        print(f"  {ch:8s} median={q[0]:.3g}  p99={q[1]:.3g}  p99.9={q[2]:.3g}  max={q[3]:.3g}  top0.1%-share={share:.2f}")

    # regime binning: input Mach + origin Mach (incl. vacuum)
    rhoL, uL, pL = 10.0 ** x[:, 1], x[:, 2], 10.0 ** x[:, 3]
    rhoR, uR, pR = 10.0 ** x[:, 4], x[:, 5], 10.0 ** x[:, 6]
    aL = jnp.sqrt(GAMMA * pL / rhoL); aR = jnp.sqrt(GAMMA * pR / rhoR)
    mach_in = jnp.maximum(jnp.abs(uL) / aL, jnp.abs(uR) / aR)

    def origin_mach(r):
        gs = GasState.from_array(r[1:])
        pstar, _ = find_pstar(gs)
        rho0, u0, e0 = sample_origin(gs, pstar)
        p0 = (GAMMA - 1.0) * rho0 * e0
        a0 = jnp.sqrt(GAMMA * p0 / jnp.maximum(rho0, 1e-300))
        return jnp.where(rho0 > 0, jnp.abs(u0) / a0, jnp.inf)

    mach0 = jax.vmap(origin_mach)(x)
    edges = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, jnp.inf]

    for mach, label in [(mach_in, "input Mach"), (mach0, "origin Mach (inf = vacuum)")]:
        print(f"\nbinned by {label}: per-bin mean loss / median / max of worst-channel |dF|/S")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (mach >= lo) & (mach < hi)
            if not bool(m.any()):
                continue
            e = jnp.max(serr, axis=1)[m]
            print(f"  [{float(lo):5.2f},{float(hi):5.2f})  {100*float(m.mean()):5.1f}%  "
                  f"mean={float(loss_pt[m].mean()):9.3e}  med={float(jnp.median(e)):8.3g}  max={float(e.max()):8.3g}")
        if mach is mach0:
            vac = jnp.isinf(mach0)
            if bool(vac.any()):
                e = jnp.max(serr, axis=1)[vac]
                print(f"  vacuum only: {100*float(vac.mean()):.1f}%  med={float(jnp.median(e)):.3g}  max={float(e.max()):.3g}")

    # Roe comparison
    fr = jax.vmap(roe_flux_x)(x)
    serr_roe = jnp.abs(fr - ft) / S
    print(f"\nRoe (no entropy fix) vs exact, training metric |dF|/S:")
    print(f"  overall loss mean((dF/S)^2): {float(jnp.mean(serr_roe**2)):.4e}")
    for i, ch in enumerate(names):
        q = jnp.percentile(serr_roe[:, i], jnp.array([50.0, 99.0, 99.9, 100.0]))
        print(f"  {ch:8s} median={q[0]:.3g}  p99={q[1]:.3g}  p99.9={q[2]:.3g}  max={q[3]:.3g}")
    loss_roe = (serr_roe**2).sum(axis=1)
    print("\nRoe binned by origin Mach (inf = vacuum):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mach0 >= lo) & (mach0 < hi)
        if not bool(m.any()):
            continue
        e = jnp.max(serr_roe, axis=1)[m]
        print(f"  [{float(lo):5.2f},{float(hi):5.2f})  {100*float(m.mean()):5.1f}%  "
              f"mean={float(loss_roe[m].mean()):9.3e}  med={float(jnp.median(e)):8.3g}  max={float(e.max()):8.3g}")
    vac = jnp.isinf(mach0)
    if bool(vac.any()):
        e = jnp.max(serr_roe, axis=1)[vac]
        print(f"  vacuum only: {100*float(vac.mean()):.1f}%  med={float(jnp.median(e)):.3g}  max={float(e.max()):.3g}")

    # optional: side-by-side with the latest NN checkpoint + worst-set overlap
    ckpt = Path(__file__).resolve().parent / "checkpoints_newfluxscale"
    if ckpt.exists():
        from checkpoint import load_latest
        from train_mat import F_pred_nomat
        F_net, step = load_latest(ckpt, _build_model)
        fp = jax.vmap(lambda r: F_pred_nomat(F_net, r))(x)
        serr_nn = jnp.abs(fp - ft) / S
        print(f"\nside-by-side with NN checkpoint (step {step}), median / p99 / max of |dF|/S:")
        for i, ch in enumerate(names):
            qh = jnp.percentile(serr[:, i], jnp.array([50.0, 99.0, 100.0]))
            qr = jnp.percentile(serr_roe[:, i], jnp.array([50.0, 99.0, 100.0]))
            qn = jnp.percentile(serr_nn[:, i], jnp.array([50.0, 99.0, 100.0]))
            print(f"  {ch:8s} HLLC {qh[0]:8.3g} {qh[1]:8.3g} {qh[2]:8.3g} | Roe {qr[0]:8.3g} {qr[1]:8.3g} {qr[2]:8.3g} | NN {qn[0]:8.3g} {qn[1]:8.3g} {qn[2]:8.3g}")
        print(f"  overall loss: HLLC {float(jnp.mean(serr**2)):.3e}   Roe {float(jnp.mean(serr_roe**2)):.3e}   NN {float(jnp.mean(serr_nn**2)):.3e}")

        # do the worst points coincide? overlap of top-0.1% sets + rank correlation
        k = max(1, N // 1000)
        loss_nn = (serr_nn**2).sum(axis=1)
        loss_hllc = (serr**2).sum(axis=1)
        sets = {nm: set(jnp.argsort(lp)[-k:].tolist())
                for nm, lp in [("NN", loss_nn), ("Roe", loss_roe), ("HLLC", loss_hllc)]}
        print(f"\nworst-0.1% set overlap (random expectation 0.1%):")
        print(f"  NN  & Roe : {100.0 * len(sets['NN'] & sets['Roe']) / k:.1f}%")
        print(f"  NN  & HLLC: {100.0 * len(sets['NN'] & sets['HLLC']) / k:.1f}%")
        print(f"  Roe & HLLC: {100.0 * len(sets['Roe'] & sets['HLLC']) / k:.1f}%")
        # rank correlation on a subsample
        idx = jax.random.permutation(jax.random.PRNGKey(0), N)[:50_000]
        def spearman(a, b):
            ra = jnp.argsort(jnp.argsort(a)).astype(jnp.float64)
            rb = jnp.argsort(jnp.argsort(b)).astype(jnp.float64)
            ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
            return float(jnp.mean(ra * rb))
        print(f"Spearman(per-point loss): NN~Roe {spearman(loss_nn[idx], loss_roe[idx]):.2f}   "
              f"NN~HLLC {spearman(loss_nn[idx], loss_hllc[idx]):.2f}   "
              f"Roe~HLLC {spearman(loss_roe[idx], loss_hllc[idx]):.2f}")


if __name__ == "__main__":
    _main(int(sys.argv[1]) if len(sys.argv) > 1 else 500_000)
