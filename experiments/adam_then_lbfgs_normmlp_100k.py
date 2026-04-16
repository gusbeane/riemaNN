"""MLP: 10k Adam warmup + 1k L-BFGS finisher on the f* residual."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureMLPNormalized

experiment = Experiment(
    model=StarPressureMLPNormalized(width=256, depth=3),
    domain=dict(
        log_rho_range=(0.0, 2.0),
        log_p_range=(0.0, 2.0),
        u_range=(-1.0, 1.0),
    ),
    seed=42,
    phases=[
        adam_cosine(n_epochs=100_000, lr=4e-3, alpha=1e-6, batch_size=2048, loss="fstar", log_every=20),
        # lbfgs(n_epochs=15, batch_size=65_536, loss="fstar", log_every=1),
    ],
)
