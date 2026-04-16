"""Normalized-MLP + Newton-step loss ((f/f')^2) on narrowed u domain."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureMLPNormalized

experiment = Experiment(
    model=StarPressureMLPNormalized(width=256, depth=3),
    domain=dict(
        log_rho_range=(0.0, 2.0),
        log_p_range=(0.0, 2.0),
        u_range=(-0.5, 0.5),
    ),
    seed=42,
    phases=[
        adam_cosine(n_epochs=10_000, lr=4e-3, alpha=1e-4, batch_size=2048, loss="newton", log_every=200),
    ],
)
