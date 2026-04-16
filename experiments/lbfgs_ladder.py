"""MLP: Adam warmup, then L-BFGS at two batch sizes."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureMLP

experiment = Experiment(
    model=StarPressureMLP(width=256, depth=3),
    domain=dict(
        log_rho_range=(-2.0, 2.0),
        log_p_range=(-2.0, 2.0),
        u_range=(-1.0, 1.0),
    ),
    seed=42,
    phases=[
        adam_cosine(n_epochs=10_000, lr=1e-3, batch_size=256, loss="fstar", log_every=200),
        lbfgs(n_epochs=1_000, batch_size=256, loss="fstar", name="lbfgs_b256"),
        lbfgs(n_epochs=500, batch_size=1024, loss="fstar", name="lbfgs_b1024"),
    ],
)
