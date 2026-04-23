"""MLP with the normalized (f/c_ref)^2 loss."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss_normalized


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(0.0, 2.0),
            log_p_range=(0.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        corner_every=200,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(4e-3, 10_000, alpha=1e-6),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss_normalized,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    ),
]
