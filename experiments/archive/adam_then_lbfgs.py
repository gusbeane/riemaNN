"""MLP: Adam warmup + two L-BFGS phases at different batch sizes."""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


experiments = [
    Experiment(
        name="main",
        model=PressureMLP(width=256, depth=3, normalize="none"),
        domain=dict(
            log_rho_range=(-2.0, 2.0),
            log_p_range=(-2.0, 2.0),
            u_range=(-1.0, 1.0),
        ),
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(1e-3, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=256,
                log_every=200,
                name="adam_cosine",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=1_000,
                loss=residual_loss,
                batch_size=256,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b256",
            ),
            Phase(
                tx=optax.lbfgs(),
                n_epochs=500,
                loss=residual_loss,
                batch_size=1_024,
                fixed_batch=True,
                is_lbfgs=True,
                name="lbfgs_b1024",
            ),
        ],
    ),
]
