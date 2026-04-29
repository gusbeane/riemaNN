"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, supervised_loss, UniformSampler


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16 # about 65k
LR = 2e-3

experiments = [
    Experiment(
        name=f"gold",
        model=PressureMLP(width=16, depth=2),
        domain=_DOMAIN,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        learning_rate=LR,
                        # optax.cosine_decay_schedule(lr, N_EPOCHS, alpha=1e-7),
                    ),
                ),
                n_epochs=N_EPOCHS,
                loss=supervised_loss,
                batch_size=BATCH_SIZE,
                sampler=UniformSampler(**_DOMAIN),
                log_every=1,
                name="adamw",
            ),
        ],
    )
]

