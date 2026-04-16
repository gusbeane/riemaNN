"""MLP: 10k Adam warmup + 1k L-BFGS finisher on the f* residual."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureMLP

experiment = Experiment(
    model=StarPressureMLP(width=256, depth=3),
    domain=dict(
        log_rho_range=(0.0, 2.0),
        log_p_range=(0.0, 2.0),
        u_range=(-1.0, 1.0),
    ),
    seed=42,
    phases=[
        adam_cosine(n_epochs=40_000, lr=1e-3, batch_size=256, loss="fstar", log_every=200),
        lbfgs(n_epochs=4_000, batch_size=1024, loss="fstar", log_every=100),
    ],
)
