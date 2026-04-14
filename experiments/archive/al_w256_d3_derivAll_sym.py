"""Residual then symmetry fine-tune: 4-phase training, w256 d3.

Phase 1: Adam (10k) with residual loss
Phase 2: L-BFGS (1k) with residual loss
Phase 3: Adam (5k) with symmetry loss (weight=0.1)
Phase 4: L-BFGS (1k) with symmetry loss
"""

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from flax.training import train_state as flax_train_state

from riemann_pinn import Experiment, losses, models, samplers, targets, training

SYMMETRY_WEIGHT = 0.1

exp = Experiment(
    name="al_w256_d3_derivAll_sym",
    model=models.StarPressureMLP(width=256, depth=3),
    target=targets.STAR_PRESSURE_LOG10,
    sampler=samplers.uniform_log,
    loss_impl=losses.symmetry_loss,
    loss_kwargs={"symmetry_weight": SYMMETRY_WEIGHT},
    optimizer={"type": "lbfgs", "memory_size": 10},
    n_epochs=17_000,
    batch_size=4096,
    seed=42,
)


def train(exp_):
    rng = jr.PRNGKey(exp_.seed)

    residual_loss_fn = losses.make_loss_fn(exp_.target, losses.residual_loss)
    sym_loss_fn = losses.make_augmented_loss_fn(
        exp_.target, losses.symmetry_loss,
        symmetry_weight=SYMMETRY_WEIGHT,
    )

    # Phase 1: Adam with residual loss (10k)
    adam_opt = training.build_optimizer({"type": "adam", "learning_rate": 1e-3})
    state = training.create_train_state(rng, exp_.model, adam_opt, batch_size_hint=256)
    step = training.make_train_step(residual_loss_fn)
    state, trace1 = training.run_training_loop(
        state, step, exp_.sampler, jr.fold_in(rng, 1),
        n_epochs=10_000, batch_size=256,
        desc=f"{exp_.name} adam+resid", log_every=2000,
    )

    # Phase 2: L-BFGS with residual loss (1k)
    lbfgs_opt = training.build_optimizer({"type": "lbfgs", "memory_size": 10})
    state = flax_train_state.TrainState.create(
        apply_fn=state.apply_fn, params=state.params, tx=lbfgs_opt,
    )
    step = training.make_lbfgs_train_step(residual_loss_fn)
    state, trace2 = training.run_training_loop(
        state, step, exp_.sampler, jr.fold_in(rng, 2),
        n_epochs=1_000, batch_size=4096,
        desc=f"{exp_.name} lbfgs+resid", log_every=100,
    )

    # Phase 3: Adam with symmetry loss (5k)
    adam_opt2 = training.build_optimizer({"type": "adam", "learning_rate": 1e-3})
    state = flax_train_state.TrainState.create(
        apply_fn=state.apply_fn, params=state.params, tx=adam_opt2,
    )
    step = training.make_train_step(sym_loss_fn)
    state, trace3 = training.run_training_loop(
        state, step, exp_.sampler, jr.fold_in(rng, 3),
        n_epochs=5_000, batch_size=256,
        desc=f"{exp_.name} adam+sym", log_every=2000,
    )

    # Phase 4: L-BFGS with symmetry loss (1k)
    lbfgs_opt2 = training.build_optimizer({"type": "lbfgs", "memory_size": 10})
    state = flax_train_state.TrainState.create(
        apply_fn=state.apply_fn, params=state.params, tx=lbfgs_opt2,
    )
    step = training.make_lbfgs_train_step(sym_loss_fn)
    state, trace4 = training.run_training_loop(
        state, step, exp_.sampler, jr.fold_in(rng, 4),
        n_epochs=1_000, batch_size=4096,
        desc=f"{exp_.name} lbfgs+sym", log_every=100,
    )

    return state, jnp.concatenate([trace1, trace2, trace3, trace4])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    exp.run_custom(train, force_retrain=args.retrain, skip_plots=args.skip_plots)
