"""Narrow/deep normalized MLP with L-BFGS on R2 quasirandom samples."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import (
    Experiment, Phase, residual_loss, r2_quasirandom,
)


experiments = [
    Experiment(
        name="lbfgs_w8_d6",
        model=PressureMLP(width=8, depth=6, normalize="geom"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(8e-3, 5_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=5_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=100_000,
                loss=residual_loss,
                batch_size=8_192,
                fixed_batch=True,
                is_lbfgs=True,
                sampler=r2_quasirandom,
                log_every=200,
                name="lbfgs",
            ),
        ],
    ),
]
