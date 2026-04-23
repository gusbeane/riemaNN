"""Width/depth/lr grid over the normalized MLP baseline (adamw_normmlp).

Same optimizer, loss, domain, seed, and phase budget as adamw_normmlp.py;
only width/depth/lr vary. Select one with --index N; outputs land in
outputs/adamw_normmlp_wdgrid/<name>/.
"""

from itertools import product

import optax

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss


_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)

_WIDTHS = [64, 128, 256]
_DEPTHS = [2, 3, 4]
_LRS    = [2e-3, 4e-3, 8e-3]


def _make(width: int, depth: int, lr: float) -> Experiment:
    return Experiment(
        name=f"w{width}_d{depth}_lr{lr}",
        model=PressureMLP(width=width, depth=depth, normalize="arith"),
        domain=_DOMAIN,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(lr, 10_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=10_000,
                loss=residual_loss,
                batch_size=2048,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    )


experiments = [_make(w, d, lr) for w, d, lr in product(_WIDTHS, _DEPTHS, _LRS)]
