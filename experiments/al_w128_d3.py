"""Adam (5k) -> L-BFGS (500) polish, w128 d3."""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets
from experiments._adam_lbfgs import adam_then_lbfgs

exp = Experiment(
    name="al_w128_d3",
    model=models.StarPressureMLP(width=128, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=5500,
    batch_size=4096,
    seed=42,
)


def train(exp_):
    return adam_then_lbfgs(exp_, )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(train, force_retrain=args.retrain, skip_plots=args.skip_plots)
