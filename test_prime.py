"""Summary stats of f'(p*) over the adam_then_lbfgs training domain."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from riemann_pinn import physics

DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)

N_PER_AXIS = 10  # 10^5 = 100_000 grid points


def build_grid():
    lr = jnp.linspace(*DOMAIN["log_rho_range"], N_PER_AXIS)
    lp = jnp.linspace(*DOMAIN["log_p_range"], N_PER_AXIS)
    u = jnp.linspace(*DOMAIN["u_range"], N_PER_AXIS)
    lrL, lpL, lrR, lpR, uRL = jnp.meshgrid(lr, lp, lr, lp, u, indexing="ij")
    return jnp.stack(
        [lrL.ravel(), lpL.ravel(), lrR.ravel(), lpR.ravel(), uRL.ravel()],
        axis=-1,
    )


def summarize(name, arr):
    a = np.asarray(arr)
    qs = np.nanpercentile(a, [0.0, 5.0, 50.0, 95.0, 100.0])
    print(
        f"{name:>6s}:  min={qs[0]: .4e}  p5={qs[1]: .4e}  "
        f"median={qs[2]: .4e}  p95={qs[3]: .4e}  max={qs[4]: .4e}"
    )


def main():
    gas_log = build_grid()
    gas_phys = physics.gas_log_to_phys(gas_log)
    print(f"grid points: {gas_log.shape[0]}")

    pstar, fres = jax.vmap(physics.find_pstar)(gas_phys)
    fprime = jax.vmap(physics.dfstar_dp)(pstar, gas_phys)

    # Report only points where the solver actually converged.
    ok = jnp.isfinite(pstar) & jnp.isfinite(fprime) & (jnp.abs(fres) < 1e-6)
    n_ok = int(ok.sum())
    print(f"converged:   {n_ok} / {gas_log.shape[0]}")

    fprime_ok = fprime[ok]

    summarize("f'",   fprime_ok)
    summarize("|f'|", jnp.abs(fprime_ok))

    # --- Where does f' get large? Show the top-N extreme points. ---
    N_SHOW = 20
    gas_log_ok = gas_log[ok]
    gas_phys_ok = gas_phys[ok]
    pstar_ok = pstar[ok]
    order = jnp.argsort(jnp.abs(fprime_ok))[::-1][:N_SHOW]

    order_np = np.asarray(order)
    gas_log_np = np.asarray(gas_log_ok)
    gas_phys_np = np.asarray(gas_phys_ok)
    pstar_np = np.asarray(pstar_ok)
    fprime_np = np.asarray(fprime_ok)

    print(f"\nTop {N_SHOW} extreme |f'| points (log-space inputs):")
    print(
        f"  {'lrL':>6s} {'lpL':>6s} {'lrR':>6s} {'lpR':>6s} {'uRL':>6s}  "
        f"{'p*':>10s}  {'pL':>10s}  {'pR':>10s}  {'fp':>12s}"
    )
    for i in order_np:
        lrL, lpL, lrR, lpR, uRL = gas_log_np[i]
        _, pL, _, pR, _ = gas_phys_np[i]
        print(
            f"  {lrL:6.2f} {lpL:6.2f} {lrR:6.2f} {lpR:6.2f} {uRL:6.2f}  "
            f"{pstar_np[i]:10.3e}  {pL:10.3e}  {pR:10.3e}  {fprime_np[i]:12.3e}"
        )

    # How rare is the blowup?
    thresholds = [1e1, 1e2, 1e4, 1e6, 1e9]
    print("\nTail mass of |f'|:")
    total = fprime_ok.size
    for t in thresholds:
        frac = float(jnp.mean(jnp.abs(fprime_ok) > t))
        print(f"  P(|f'| > {t:.0e}) = {frac:.4%}  ({int(frac * total)} / {total})")


if __name__ == "__main__":
    main()
