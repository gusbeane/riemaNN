"""Sweep AdamW learning rate on the gold-style architecture."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss, mse_log_loss
from typing import Callable

_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16
LR_STAGE1 = 0.06

def _stage(name: str, loss: Callable, lr: float) -> Stage:
    return Stage(
        name=name,
        model=PressureMLP(width=16, depth=2),
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(learning_rate=lr),
                ),
                n_epochs=N_EPOCHS,
                loss=loss,
                batch_size=BATCH_SIZE,
                sampler=UniformSampler(**_DOMAIN),
                log_every=1,
                name="adamw",
            ),
        ],
    )

experiments = [
    Experiment(
        name=f"lr{lr}",
        seed=42,
        domain=_DOMAIN,
        stages=[
            _stage("base", mse_log_loss, LR_STAGE1),
            _stage("correction", mse_loss, lr),
        ],
    )
    for lr in [1e-3, 4e-3, 1e-2, 4e-2, 6e-2, 8e-2]
]
