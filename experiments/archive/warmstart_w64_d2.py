"""w64, d2: 1k epochs of supervised warm-start (MSE vs find_pstar) then
4k epochs of residual fine-tuning.

The hypothesis is that the residual loss gets stuck in a bad basin
because 20% of samples have Newton divergence and the network never
learns a reasonable log10(p*) to start with. Supervised pre-training
on the samples Newton *does* converge for should give the residual
phase a much better initialization.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from riemann_pinn import Experiment, losses, models, samplers, targets, training

N_EPOCHS_SUPERVISED = 1_000
N_EPOCHS_RESIDUAL = 4_000

exp = Experiment(
    name="archive/warmstart_w64_d2",
    model=models.StarPressureMLP(width=64, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.residual_loss,  # reflects the final phase in config.json
    optimizer={"type": "adam", "learning_rate": 1e-3},
    n_epochs=N_EPOCHS_SUPERVISED + N_EPOCHS_RESIDUAL,
    batch_size=256,
    seed=42,
)


def two_phase_train(exp: Experiment):
    state = exp._build_template_state()

    # Phase 1: supervised (log10 MSE against Newton-iterated p*).
    sup_loss_fn = losses.make_loss_fn(exp.target, losses.supervised_loss)
    sup_step = training.make_train_step(sup_loss_fn)
    rng_sup = jr.fold_in(jr.PRNGKey(exp.seed), 1)
    state, trace_sup = training.run_training_loop(
        state,
        sup_step,
        exp.sampler,
        rng_sup,
        n_epochs=N_EPOCHS_SUPERVISED,
        batch_size=exp.batch_size,
        desc=f"{exp.name} phase 1 (supervised)",
    )

    # Phase 2: residual fine-tuning.
    res_loss_fn = losses.make_loss_fn(exp.target, losses.residual_loss)
    res_step = training.make_train_step(res_loss_fn)
    rng_res = jr.fold_in(jr.PRNGKey(exp.seed), 2)
    state, trace_res = training.run_training_loop(
        state,
        res_step,
        exp.sampler,
        rng_res,
        n_epochs=N_EPOCHS_RESIDUAL,
        batch_size=exp.batch_size,
        desc=f"{exp.name} phase 2 (residual)",
    )

    loss_trace = jnp.concatenate([trace_sup, trace_res])
    return state, loss_trace


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(
        two_phase_train,
        force_retrain=args.retrain,
        skip_plots=args.skip_plots,
    )
