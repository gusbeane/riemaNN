"""Width/depth grid over the normalized MLP baseline (adamw_normmlp).

Same optimizer, loss, domain, seed, and phase budget as `adamw_normmlp.py`; only
the `StarPressureMLPNormalized(width, depth)` varies. Select one with
``--index N``; outputs land in ``outputs/adamw_normmlp_wdgrid/<name>/``.
"""

from itertools import product

from riemann_pinn.experiment import Experiment, adam_cosine
from riemann_pinn.model import StarPressureMLPNormalized

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-1.0, 1.0),
)

_WIDTHS = [64, 128, 256]
_DEPTHS = [2, 3, 4]
_LRS = [2e-3, 4e-3, 8e-3]


def _make(width: int, depth: int, lr: float) -> Experiment:
    return Experiment(
        model=StarPressureMLPNormalized(width=width, depth=depth),
        domain=_DOMAIN,
        seed=42,
        name=f"w{width}_d{depth}_lr{lr}",
        phases=[
            adam_cosine(
                n_epochs=10_000, lr=lr, alpha=0.0, batch_size=2048,
                loss="fstar", log_every=200,
            ),
        ],
    )


experiments = [_make(w, d, lr) for w, d, lr in product(_WIDTHS, _DEPTHS, _LRS)]
