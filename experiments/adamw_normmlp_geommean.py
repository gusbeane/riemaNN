"""Arithmetic vs geometric mean for the normalized-MLP reference scales.

Base config is w64_d2_lr0.008 of adamw_normmlp_wdgrid.py; only the
normalize= mode changes ("arith" vs "geom"). Geometric is the arithmetic
mean in log space and is antisymmetric under L<->R, which may make the
learning problem easier.
"""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)


def _phases():
    return [
        Phase(
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    optax.cosine_decay_schedule(8e-3, 10_000, alpha=0.0),
                    weight_decay=1e-4,
                ),
            ),
            n_epochs=10_000,
            loss=residual_loss,
            batch_size=2048,
            log_every=200,
            name="adam_cosine",
        ),
    ]


experiments = [
    Experiment(
        name="arithmetic",
        model=PressureMLP(width=64, depth=2, normalize="arith"),
        domain=_DOMAIN, seed=42, phases=_phases(),
    ),
    Experiment(
        name="geometric",
        model=PressureMLP(width=64, depth=2, normalize="geom"),
        domain=_DOMAIN, seed=42, phases=_phases(),
    ),
]
