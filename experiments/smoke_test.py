"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax

from riemann_pinn.data import UniformSampler
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, Stage, mse_loss


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.9),
)


experiments = [
    Experiment(
        name="smoke_test",
        seed=42,
        domain=_DOMAIN,
        stages=[
            Stage(
                name="main",
                model=PressureMLP(width=64, depth=2),
                phases=[
                    Phase(
                        tx=optax.chain(
                            optax.clip_by_global_norm(1.0),
                            optax.adamw(
                                optax.cosine_decay_schedule(4e-3, 1_000, alpha=0.0),
                                weight_decay=1e-4,
                            ),
                        ),
                        n_epochs=1_000,
                        loss=mse_loss,
                        batch_size=2048,
                        sampler=UniformSampler(**_DOMAIN),
                        log_every=100,
                        name="adam_cosine",
                    ),
                ],
            ),
        ],
    ),
]
