"""Derivative loss (uRL + rho): Adam (10k) -> L-BFGS (1k), w256 d3.

Weights are set after hyperparameter sweeps.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets
from experiments._adam_lbfgs import adam_then_lbfgs_with_loss_fn

DERIV_URL_WEIGHT = 0.1   # tuned by individual sweep
DERIV_RHO_WEIGHT = 0.01  # tuned by individual sweep

exp = Experiment(
    name="al_w256_d3_derivAll",
    model=models.StarPressureMLP(width=256, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.derivative_loss,
    loss_kwargs={
        "deriv_uRL_weight": DERIV_URL_WEIGHT,
        "deriv_rho_weight": DERIV_RHO_WEIGHT,
    },
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=11_000,
    batch_size=4096,
    seed=42,
)


def train(exp_):
    loss_fn = losses.make_augmented_loss_fn(
        exp_.target, losses.derivative_loss,
        deriv_uRL_weight=DERIV_URL_WEIGHT,
        deriv_rho_weight=DERIV_RHO_WEIGHT,
    )
    return adam_then_lbfgs_with_loss_fn(
        exp_, loss_fn, adam_epochs=10_000, lbfgs_epochs=1_000
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(train, force_retrain=args.retrain, skip_plots=args.skip_plots)
