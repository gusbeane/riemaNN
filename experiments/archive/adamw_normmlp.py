"""Normalized MLP: 10k Adam-cosine baseline on the f* residual."""

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
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
