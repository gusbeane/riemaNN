"""Symmetric architecture: Adam (10k) -> L-BFGS (1k), w256 d3."""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets
from experiments._adam_lbfgs import adam_then_lbfgs

exp = Experiment(
    name="al_w256_d3_symarch",
    model=models.SymmetricMLP(width=256, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=11_000,
    batch_size=4096,
    seed=42,
)


def train(exp_):
    return adam_then_lbfgs(exp_, adam_epochs=10_000, lbfgs_epochs=1_000)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(train, force_retrain=args.retrain, skip_plots=args.skip_plots)
