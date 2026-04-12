"""L-BFGS optimizer, w128, d3."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.random as jr

from riemann_pinn import Experiment, losses, models, samplers, targets, training

exp = Experiment(
    name="lbfgs_w128_d3",
    model=models.StarPressureMLP(width=128, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=500,
    batch_size=2048,
    seed=42,
)


def lbfgs_train(exp_: Experiment):
    state = exp_._build_template_state()
    loss_fn = losses.make_loss_fn(exp_.target, exp_.loss_impl, **exp_.loss_kwargs)
    train_step = training.make_lbfgs_train_step(loss_fn)
    rng = jr.fold_in(jr.PRNGKey(exp_.seed), 1)
    state, loss_trace = training.run_training_loop(
        state, train_step, exp_.sampler, rng,
        n_epochs=exp_.n_epochs, batch_size=exp_.batch_size, desc=exp_.name,
        log_every=100,
    )
    return state, loss_trace


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(lbfgs_train, force_retrain=args.retrain, skip_plots=args.skip_plots)
