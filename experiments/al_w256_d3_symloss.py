"""Symmetry loss: Adam (10k) -> L-BFGS (1k), w256 d3.

symmetry_weight is set after hyperparameter sweep.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets
from experiments._adam_lbfgs import adam_then_lbfgs_with_loss_fn

SYMMETRY_WEIGHT = 0.1  # tuned by sweep: best med_rel_err among [0.001, 0.01, 0.1, 1.0, 10.0]

exp = Experiment(
    name="al_w256_d3_symloss",
    model=models.StarPressureMLP(width=256, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.symmetry_loss,
    loss_kwargs={"symmetry_weight": SYMMETRY_WEIGHT},
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=11_000,
    batch_size=4096,
    seed=42,
)


def train(exp_):
    loss_fn = losses.make_augmented_loss_fn(
        exp_.target, losses.symmetry_loss,
        symmetry_weight=SYMMETRY_WEIGHT,
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
