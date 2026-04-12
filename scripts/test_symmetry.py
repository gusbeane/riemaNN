"""Test how symmetric each trained network is under L/R swap.

For each experiment, generates holdout samples and compares
  p*(qL, qR, uRL)  vs  p*(qR, qL, -uRL)

A perfectly symmetric solver would give identical results.

Usage:
    venv/bin/python -m scripts.test_symmetry
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from riemann_pinn import samplers
from riemann_pinn.experiment import Experiment


def swap_inputs(gas_states_log: jnp.ndarray) -> jnp.ndarray:
    """Swap L/R states and flip uRL sign."""
    return jnp.concatenate([
        gas_states_log[:, 2:4],
        gas_states_log[:, 0:2],
        -gas_states_log[:, 4:5],
    ], axis=-1)


def measure_asymmetry(state, gas_states_log):
    """Return per-sample |log10(p_fwd) - log10(p_swap)|."""
    raw_fwd = state.apply_fn({"params": state.params}, gas_states_log)
    raw_swap = state.apply_fn({"params": state.params}, swap_inputs(gas_states_log))

    # For log10-target models, raw output is log10(p*), so decode as 10**raw
    p_fwd = 10.0 ** raw_fwd
    p_swap = 10.0 ** raw_swap

    # Asymmetry in log space
    log_asym = jnp.abs(jnp.log10(jnp.maximum(p_fwd, 1e-30))
                       - jnp.log10(jnp.maximum(p_swap, 1e-30)))
    # Relative asymmetry
    rel_asym = jnp.abs(p_fwd - p_swap) / jnp.maximum(
        0.5 * (jnp.abs(p_fwd) + jnp.abs(p_swap)), 1e-30
    )
    return np.asarray(log_asym), np.asarray(rel_asym)


def main():
    rng = jr.PRNGKey(999)
    gas_states_log = samplers.uniform_log(rng, 20_000)

    experiments = [
        "al_w256_d3",
        "al_w256_d3_symloss",
        "al_w256_d3_symarch",
        "al_w256_d3_derivAll_sym",
        "al_w256_d3_p0input",
        "al_w256_d3_newton",
        "al_w256_d3_derivAll",
    ]

    print(f"{'experiment':<30s}  {'med |Δlog p|':>12s}  {'p95 |Δlog p|':>12s}  "
          f"{'med rel':>10s}  {'p95 rel':>10s}  {'max rel':>10s}")
    print("-" * 100)

    for name in experiments:
        try:
            result = Experiment.load(name)
        except Exception as e:
            print(f"{name:<30s}  LOAD FAILED: {e}")
            continue

        log_asym, rel_asym = measure_asymmetry(result.state, gas_states_log)

        print(f"{name:<30s}  "
              f"{np.median(log_asym):12.2e}  "
              f"{np.percentile(log_asym, 95):12.2e}  "
              f"{np.median(rel_asym):10.2e}  "
              f"{np.percentile(rel_asym, 95):10.2e}  "
              f"{np.max(rel_asym):10.2e}")


if __name__ == "__main__":
    main()
