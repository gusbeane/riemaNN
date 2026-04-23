"""Tiny experiment for smoke-testing the pipeline (few epochs)."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=32, depth=2, normalize="none"),
        domain=dict(
            log_rho_range=(-2.0, 2.0),
            log_p_range=(-2.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=0,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(1e-3, 20, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=20,
                loss=residual_loss,
                batch_size=64,
                log_every=10,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=10,
                loss=residual_loss,
                batch_size=64,
                fixed_batch=True,
                is_lbfgs=True,
                log_every=5,
                name="lbfgs",
            ),
        ],
    ),
]
