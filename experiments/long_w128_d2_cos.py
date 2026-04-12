"""Long w128, d2, but with cosine decaying learning rate.

Longer training didn't help, but I think lr is too high and so its having
trouble finding the minimum.
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets

import optax
optimizer = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=5_000, alpha=1e-2)

exp = Experiment(
    name="long_w128_d2_cos",
    model=models.StarPressureMLP(width=128, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer=optimizer,
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
