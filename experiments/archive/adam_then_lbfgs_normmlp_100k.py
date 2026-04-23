"""Normalized MLP: 100k Adam-cosine, no L-BFGS."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
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
                        optax.cosine_decay_schedule(4e-3, 100_000, alpha=1e-6),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=100_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=20,
                name="adam_cosine",
            ),
        ],
    ),
]
