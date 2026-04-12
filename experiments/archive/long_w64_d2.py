"""Baseline w64, d2, but trained for 20k epochs instead of 5k.

Sanity check on whether the baseline is actually converged or just
undertrained. If long_w64_d2 is much better than baseline_w64_d2, the
original 5k budget was the bottleneck.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets

exp = Experiment(
    name="archive/long_w64_d2",
    model=models.StarPressureMLP(width=64, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer={"type": "adam", "learning_rate": 1e-3},
    n_epochs=20_000,
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
