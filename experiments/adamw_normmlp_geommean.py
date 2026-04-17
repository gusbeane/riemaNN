"""Arithmetic vs geometric mean for the normalized-MLP reference scales.

Base config is the w64_d2_lr0.008 cell of adamw_normmlp_wdgrid.py. Only the
reference-scale definition changes:

  arithmetic:  rho_ref = 0.5 * (rhoL + rhoR),   p_ref = 0.5 * (pL + pR)
  geometric:   rho_ref = sqrt(rhoL * rhoR),      p_ref = sqrt(pL * pR)

The geometric variant is the arithmetic mean in log space and is antisymmetric
under L<->R, which may make the learning problem easier.
"""

from riemann_pinn.experiment import Experiment, adam_cosine
from riemann_pinn.model import (
    StarPressureMLPNormalized,
    StarPressureMLPNormalizedGeom,
)

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)


def _phases():
    return [
        adam_cosine(
            n_epochs=10_000, lr=8e-3, alpha=0.0, batch_size=2048,
            loss="fstar", log_every=200,
        ),
    ]


experiments = [
    Experiment(
        model=StarPressureMLPNormalized(width=64, depth=2),
        domain=_DOMAIN, seed=42, name="arithmetic", phases=_phases(),
    ),
    Experiment(
        model=StarPressureMLPNormalizedGeom(width=64, depth=2),
        domain=_DOMAIN, seed=42, name="geometric", phases=_phases(),
    ),
]
