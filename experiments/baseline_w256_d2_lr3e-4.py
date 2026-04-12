"""Baseline w256, d2, but with Adam learning rate 3e-4 instead of 1e-3.

Width-LR coupling check. The original baseline sweep ran every width
at lr=1e-3, which is probably too aggressive for larger networks.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets

exp = Experiment(
    name="baseline_w256_d2_lr3e-4",
    model=models.StarPressureMLP(width=256, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer={"type": "adam", "learning_rate": 3e-4},
    n_epochs=5_000,
    batch_size=256,
    seed=42,
)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run(force_retrain=args.retrain, skip_plots=args.skip_plots)
