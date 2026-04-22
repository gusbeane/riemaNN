"""Normalized MLP + Newton-step loss ((f/f')^2) on narrowed u domain."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss_newton


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="arith"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-0.5, 0.5),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=1e-4),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss_newton,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
