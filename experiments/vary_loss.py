"""Sweep AdamW learning rate on the gold-style architecture."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss, mse_log_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16
LR = 2e-3

experiments = [
    Experiment(
        name=f"{loss.__name__}",
        seed=42,
        domain=_DOMAIN,
        stages=[
            Stage(
                name="main",
                model=PressureMLP(width=16, depth=2),
                phases=[
                    Phase(
                        tx=optax.chain(
                            optax.clip_by_global_norm(1.0),
                            optax.adamw(learning_rate=LR),
                        ),
                        n_epochs=N_EPOCHS,
                        loss=loss,
                        batch_size=BATCH_SIZE,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=1,
                        name="adamw",
                    ),
                ],
            ),
        ],
    )
    for loss in [mse_loss, mse_log_loss]
]
