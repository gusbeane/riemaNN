"""Residual-correction network stacked on the `adamw_normmlp_geom_widen` wide run.

The primary (frozen) predicts p*_primary; the correction network takes the
log-space gas state plus log10 p*_primary and emits a signed log10 correction
delta, yielding p*_final = p*_primary * 10^delta. The primary's params live
outside the TrainState and are loaded from its existing checkpoint, so the
correction can be trained without retraining the primary.

Three variants:
  unsupervised:             loss="fstar" — minimize f(p*_final)^2.
  supervised:               loss="pstar" — minimize (p*_final - p*_true)^2.
  unsupervised_normalized:  scaled residual loss (f/eps)^2, where eps is the
                            primary's median |f(p*_primary)| over the domain.
                            Computed once at module import.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from riemann_pinn import physics
from riemann_pinn.experiment import (
    Experiment, PrimarySpec, adam_cosine, load_primary,
)
from riemann_pinn.model import PstarLogCorrectionMLP
from riemann_pinn.train import uniform_log

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)

_PRIMARY = PrimarySpec(
    experiment_path=Path("experiments/adamw_normmlp_geom_widen.py"),
    index=1,  # "wide" variant
)


def _primary_max_fstar(n_samples: int = 20_000, seed: int = 123) -> float:
    """Median |f(p*_primary)| across a uniform draw over _DOMAIN.

    Used as the normalization constant eps in the normalized loss variant.
    """
    primary_model, primary_params = load_primary(_PRIMARY)
    x = uniform_log(jr.PRNGKey(seed), n_samples, **_DOMAIN)
    pstar = primary_model.apply({"params": primary_params}, x)
    phys_states = physics.gas_log_to_phys(x)
    f = jax.vmap(physics.fstar)(pstar, phys_states)
    return float(jnp.max(jnp.abs(f)))


_EPS = _primary_max_fstar()


def _residual_loss_scaled(params, apply_fn, gas_states_log):
    """Mean of (f(p*) / eps)^2, with eps = primary's median |f|."""
    pstar = apply_fn({"params": params}, gas_states_log)
    phys_states = physics.gas_log_to_phys(gas_states_log)
    f = jax.vmap(physics.fstar)(pstar, phys_states)
    val = (f / _EPS) ** 2
    loss = jnp.mean(val)
    return loss, {"loss/scaled_fstar": loss}


def _make(name: str, loss) -> Experiment:
    return Experiment(
        model=PstarLogCorrectionMLP(width=64, depth=2),
        domain=_DOMAIN,
        primary=_PRIMARY,
        phases=[
            adam_cosine(
                n_epochs=40_000, lr=8e-3, alpha=0.0, batch_size=2048,
                loss=loss, log_every=200,
            ),
        ],
        seed=42,
        name=name,
    )


experiments = [
    _make("unsupervised", "fstar"),
    _make("supervised", "pstar"),
    _make("unsupervised_normalized", _residual_loss_scaled),
]
