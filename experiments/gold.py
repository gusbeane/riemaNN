"""Gold reference network: single-stage AdamW training in the 3D delta-space."""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


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
        name="gold",
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
                        loss=mse_loss,
                        batch_size=BATCH_SIZE,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=1,
                        name="adamw",
                    ),
                ],
            ),
        ],
    ),
]
