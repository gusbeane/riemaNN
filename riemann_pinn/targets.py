"""Target abstraction: decouples network output shape from loss construction.

A `Target` bundles two stateless functions:

- `decode(gas_states_log, raw_out)` turns the network's raw output into a
  dict of named physical predictions (e.g. `{"pstar": ...}`).
- `residuals(preds, gas_states_phys)` computes a dict of named physical
  residuals that should be zero at the true solution (e.g.
  `{"fstar": fstar(pstar, state)}`).

Losses in `losses.py` consume these dicts, which means you can swap in a
new prediction/residual pair (e.g. "predict (p*, u*), constrain both")
without touching the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from . import physics


@dataclass(frozen=True)
class Target:
    name: str
    # (gas_states_log, raw_model_out) -> {pred_name: array(B,)}
    decode: Callable[[jnp.ndarray, jnp.ndarray], dict[str, jnp.ndarray]]
    # (preds, gas_states_phys) -> {residual_name: array(B,)}
    residuals: Callable[[dict[str, jnp.ndarray], jnp.ndarray], dict[str, jnp.ndarray]]


def _star_pressure_log10_decode(gas_states_log, raw_out):
    # raw_out is expected shape (B,) — StarPressureMLP squeezes when output_dim=1.
    return {"pstar": 10.0 ** raw_out}


def _star_pressure_log10_residuals(preds, gas_states_phys):
    pstar = preds["pstar"]
    return {"fstar": jax.vmap(physics.fstar)(pstar, gas_states_phys)}


STAR_PRESSURE_LOG10 = Target(
    name="star_pressure_log10",
    decode=_star_pressure_log10_decode,
    residuals=_star_pressure_log10_residuals,
)


TARGET_REGISTRY: dict[str, Target] = {
    STAR_PRESSURE_LOG10.name: STAR_PRESSURE_LOG10,
}
