"""Test find_pstar convergence on random samples from the training domain."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from riemann_pinn import physics, samplers


def test_find_pstar(n: int = 100_000, seed: int = 0):
    rng = jr.PRNGKey(seed)
    gas_log = samplers.uniform_log(rng, n)
    gas_phys = physics.gas_log_to_phys(gas_log)

    pstar, fstar_final = jax.vmap(physics.find_pstar)(gas_phys)

    converged = jnp.abs(fstar_final) < 1e-6
    positive = pstar > 0
    finite = jnp.isfinite(pstar) & jnp.isfinite(fstar_final)
    good = converged & positive & finite
    n_fail = n - int(jnp.sum(good))

    print(f"N = {n}")
    print(
        f"converged (|f| < 1e-6): {int(jnp.sum(converged))}/{n}"
        f" ({100 * float(jnp.mean(converged)):.2f}%)"
    )
    print(
        f"positive p*:            {int(jnp.sum(positive))}/{n}"
        f" ({100 * float(jnp.mean(positive)):.2f}%)"
    )
    print(
        f"finite:                 {int(jnp.sum(finite))}/{n}"
        f" ({100 * float(jnp.mean(finite)):.2f}%)"
    )
    print(
        f"all good:               {n - n_fail}/{n}"
        f" ({100 * (n - n_fail) / n:.2f}%)"
    )
    print(f"failures:               {n_fail}/{n} ({100 * n_fail / n:.2f}%)")

    # Failure breakdown
    not_converged = ~converged & finite
    not_positive = (pstar <= 0) & finite
    not_finite_p = ~jnp.isfinite(pstar)
    not_finite_f = ~jnp.isfinite(fstar_final)
    print(f"\nFailure breakdown:")
    print(f"  not converged (|f|>=1e-6, finite): {int(jnp.sum(not_converged))}")
    print(f"  p* <= 0 (finite):                  {int(jnp.sum(not_positive))}")
    print(f"  p* not finite:                     {int(jnp.sum(not_finite_p))}")
    print(f"  f* not finite:                     {int(jnp.sum(not_finite_f))}")

    # Show sample failures
    fail_mask = ~good
    n_show = min(10, n_fail)
    if n_show > 0:
        fail_idx = jnp.where(fail_mask, size=n_show)[0]
        print(f"\nSample failures (first {n_show}):")
        print(
            f"{'idx':>6} {'log_rhoL':>9} {'log_pL':>9} {'log_rhoR':>9}"
            f" {'log_pR':>9} {'uRL':>9} | {'p*':>12} {'|f|':>12}"
        )
        for i in fail_idx:
            gl = gas_log[i]
            print(
                f"{int(i):6d} {float(gl[0]):9.3f} {float(gl[1]):9.3f}"
                f" {float(gl[2]):9.3f} {float(gl[3]):9.3f}"
                f" {float(gl[4]):9.3f} | {float(pstar[i]):12.4e}"
                f" {float(jnp.abs(fstar_final[i])):12.4e}"
            )

    return n_fail


if __name__ == "__main__":
    test_find_pstar()
