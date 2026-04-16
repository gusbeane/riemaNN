"""Deep-Set architecture trained with supervised p* loss."""

from riemann_pinn.experiment import Experiment, adam_cosine, lbfgs
from riemann_pinn.model import StarPressureDS

experiment = Experiment(
    model=StarPressureDS(
        phi_width=256, phi_depth=3, phi_output_dim=16,
        rho_width=256, rho_depth=3,
    ),
    domain=dict(
        log_rho_range=(-2.0, 2.0),
        log_p_range=(-2.0, 2.0),
        u_range=(-1.0, 1.0),
    ),
    seed=42,
    phases=[
        adam_cosine(n_epochs=10_000, lr=1e-3, batch_size=256, loss="pstar", log_every=200),
        lbfgs(n_epochs=1_000, batch_size=256, loss="pstar"),
    ],
)
