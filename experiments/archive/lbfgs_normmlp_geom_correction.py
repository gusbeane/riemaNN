"""Adam warmup -> L-BFGS residual correction on the `adamw_normmlp_geom_widen` wide run.

Two phases, both using R2-quasirandom sampling with `loss="fstar"`:
  1. Adam (cosine LR, 10000 epochs, batch 2048) -- warm start from random init.
  2. L-BFGS (10000 epochs, full batch 16384) -- the `lbfgs` factory defaults
     `split_key_every = n_epochs + 1`, so `run_training_loop` only splits the
     batch key at epoch 0 and the same batch is reused for all 10000 epochs.

List form with a single entry so additional variants can be appended later
without breaking paths.
"""

from pathlib import Path

from riemann_pinn.experiment import Experiment, PrimarySpec, adam_cosine, lbfgs
from riemann_pinn.model import PstarLogCorrectionMLP

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)

_PRIMARY = PrimarySpec(
    experiment_path=Path("experiments/adamw_normmlp_geom_widen.py"),
    index=1,  # "wide" variant
)


experiments = [
    Experiment(
        model=PstarLogCorrectionMLP(width=64, depth=2),
        domain=_DOMAIN,
        primary=_PRIMARY,
        phases=[
            adam_cosine(
                n_epochs=10_000, lr=8e-3, alpha=0.0, batch_size=2048,
                sampler="r2", loss="fstar", log_every=200,
                name="adam_warmup",
            ),
            lbfgs(
                n_epochs=10_000, batch_size=16_384,
                sampler="r2", loss="fstar", log_every=200,
            ),
        ],
        seed=42,
        name="adam_then_lbfgs",
    ),
]
