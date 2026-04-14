"""p0 from derivAll network: Adam (10k) -> L-BFGS (1k), w256 d3.

Uses the pretrained al_w256_d3_derivAll network's prediction as the
6th input (instead of the two-rarefaction approximation).
"""

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, losses, models, samplers, targets
from riemann_pinn.experiment import Experiment as _Exp
from experiments._adam_lbfgs import adam_then_lbfgs

# --- Load frozen derivAll network as a guess function ---
_derivAll = _Exp.load("al_w256_d3_derivAll")
_derivAll_params = _derivAll.state.params
_derivAll_apply = _derivAll.state.apply_fn


def _derivAll_guess(gas_states_log):
    """Frozen derivAll network: (B, 5) log-space -> (B,) log10(p*) guess."""
    return _derivAll_apply({"params": _derivAll_params}, gas_states_log)


exp = Experiment(
    name="al_w256_d3_p0derivAll",
    model=models.GuessInputMLP(guess_fn=_derivAll_guess, width=256, depth=3),
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
