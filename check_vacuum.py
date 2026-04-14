"""Sample from the training domain and flag any vacuum states."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from riemann_pinn.physics import GAMMA, MU, gas_log_to_phys
from riemann_pinn.train import uniform_log

rng = jr.PRNGKey(0)
gas_log = uniform_log(rng, 1000, log_rho_range=(0.0, 2.0), log_p_range=(-2.0, 2.0), u_range=(-1.0, 1.0))
gas_phys = gas_log_to_phys(gas_log)

rhoL, pL, rhoR, pR, uRL = gas_phys.T
cL = jnp.sqrt(GAMMA * pL / rhoL)
cR = jnp.sqrt(GAMMA * pR / rhoR)

# Vacuum when uRL >= (cL + cR) / MU  (no positive p* root)
threshold = (cL + cR) / MU
is_vacuum = uRL >= threshold

n_vac = int(jnp.sum(is_vacuum))
print(f"Vacuum states: {n_vac} / 1000\n")

if n_vac > 0:
    idxs = jnp.where(is_vacuum)[0]
    print(f"{'i':>4}  {'log_rhoL':>8} {'log_pL':>8} {'log_rhoR':>8} {'log_pR':>8} {'uRL':>8}  {'(cL+cR)/MU':>10}")
    for i in idxs:
        row = gas_log[i]
        print(f"{int(i):4d}  {float(row[0]):8.3f} {float(row[1]):8.3f} {float(row[2]):8.3f} {float(row[3]):8.3f} {float(row[4]):8.3f}  {float(threshold[i]):10.4f}")
