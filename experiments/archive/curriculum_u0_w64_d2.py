"""w64, d2 curriculum: 2k epochs with uRL=0 (static contact discontinuity
regime) then 3k epochs with the full uRL range.

The u=0 subproblem has a symmetry (p* is purely determined by the
density/pressure ratio across the interface) and is considerably
easier to fit than the full problem with arbitrary velocity jumps.
Training on it first should put the network somewhere sensible in
weight space before the harder general case is introduced.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from riemann_pinn import Experiment, losses, models, samplers, targets, training

N_EPOCHS_EASY = 2_000
N_EPOCHS_FULL = 3_000

exp = Experiment(
    name="archive/curriculum_u0_w64_d2",
    model=models.StarPressureMLP(width=64, depth=2),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,  # phase 2 sampler; also the config-level default
    loss_impl=losses.residual_loss,
    optimizer={"type": "adam", "learning_rate": 1e-3},
    n_epochs=N_EPOCHS_EASY + N_EPOCHS_FULL,
    batch_size=256,
    seed=42,
)

# Phase 1 sampler: same uniform_log but with uRL pinned to zero.
u0_sampler = samplers.uniform_log_factory(u_range=(0.0, 0.0))


def curriculum_train(exp: Experiment):
    state = exp._build_template_state()
    loss_fn = losses.make_loss_fn(exp.target, losses.residual_loss)
    train_step = training.make_train_step(loss_fn)

    # Phase 1: easy regime, uRL = 0.
    rng_easy = jr.fold_in(jr.PRNGKey(exp.seed), 1)
    state, trace_easy = training.run_training_loop(
        state,
        train_step,
        u0_sampler,
        rng_easy,
        n_epochs=N_EPOCHS_EASY,
        batch_size=exp.batch_size,
        desc=f"{exp.name} phase 1 (uRL=0)",
    )

    # Phase 2: full problem.
    rng_full = jr.fold_in(jr.PRNGKey(exp.seed), 2)
    state, trace_full = training.run_training_loop(
        state,
        train_step,
        exp.sampler,
        rng_full,
        n_epochs=N_EPOCHS_FULL,
        batch_size=exp.batch_size,
        desc=f"{exp.name} phase 2 (full)",
    )

    loss_trace = jnp.concatenate([trace_easy, trace_full])
    return state, loss_trace


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(
        curriculum_train,
        force_retrain=args.retrain,
        skip_plots=args.skip_plots,
    )
