"""w64, d2, with Huber loss on residuals instead of squared residuals.

Hypothesis: the baseline's p95 residual is huge while the median is
small, meaning a few outlier gas states dominate gradients in the
squared loss. Huber caps the gradient magnitude for samples with
|residual| > delta, letting the easy bulk continue improving without
being pulled around by the hard tail.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets

exp = Experiment(
    name="huber_w64_d2",
    model=models.StarPressureMLP(width=64, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.huber_residual_loss,
    loss_kwargs={"delta": 0.1},  # baseline p95 |f| is ~0.5, median is ~0.12
    optimizer={"type": "adam", "learning_rate": 1e-3},
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
