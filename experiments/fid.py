"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, supervised_loss, uniform


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 100
BATCH_SIZE = 2**16 # about 65k

experiments = [
    Experiment(
        name=f"lr{lr}",
        model=PressureMLP(width=16, depth=2),
        domain=_DOMAIN,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(lr, N_EPOCHS, alpha=1e-7),
                    ),
                ),
                n_epochs=N_EPOCHS,
                loss=supervised_loss,
                batch_size=BATCH_SIZE,
                sampler=uniform,
                log_every=1,
                name="adam_cosine",
            ),
        ],
    ) for lr in [1e-4, 2e-4, 4e-4, 1e-3, 2e-3, 4e-3]
]

