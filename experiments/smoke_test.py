"""Tiny experiment for smoke-testing the pipeline (few epochs)."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureMLP

experiment = Experiment(
    model=StarPressureMLP(width=32, depth=2),
    domain=dict(
        log_rho_range=(-2.0, 2.0),
        log_p_range=(-2.0, 2.0),
        u_range=(-1.0, 1.0),
    ),
    seed=0,
    phases=[
        adam_cosine(n_epochs=20, lr=1e-3, batch_size=64, loss="fstar", log_every=10),
        lbfgs(n_epochs=10, batch_size=64, loss="fstar", log_every=5),
    ],
)
