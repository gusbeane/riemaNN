"""Two-stage gold pipeline: stage 0 predicts p* directly, stage 1 predicts
the multiplicative correction p*_true / p*_stage0. Default Stage
make_targets / combine handle the residual chaining."""

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


def _stage(name: str) -> Stage:
    return Stage(
        name=name,
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
    )


experiments = [
    Experiment(
        name="gold_stage2",
        seed=42,
        domain=_DOMAIN,
        stages=[_stage("base"), _stage("correction")],
    ),
]
