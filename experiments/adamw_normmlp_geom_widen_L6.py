"""Mixed p*-MSE + L6 penalty atop adamw_normmlp_geom_widen.

Same model, domain (wide train_domain), and budget as the widen experiment;
sweeps the l6 weight in a mean_sq + l6 * mean_abs6 supervised loss.
"""

from math import prod

import jax
import jax.numpy as jnp
import optax

from riemann_pinn import physics
from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase


_BASE_BATCH_SIZE = 2048

_DOMAIN = dict(
    log_rho_range=(0.0, 2.0),
    log_p_range=(0.0, 2.0),
    u_range=(-0.5, 0.5),
)
_WIDE_TRAIN_DOMAIN = dict(
    log_rho_range=(-0.2, 2.2),
    log_p_range=(-0.2, 2.2),
    u_range=(-0.6, 0.6),
)


def _volume(dom: dict) -> float:
    widths = {k: v[1] - v[0] for k, v in dom.items()}
    return prod([
        widths["log_rho_range"], widths["log_p_range"],
        widths["log_rho_range"], widths["log_p_range"],
        widths["u_range"],
    ])


def _scaled_batch(train_domain: dict) -> int:
    ratio = _volume(train_domain) / _volume(_DOMAIN)
    return int(round(_BASE_BATCH_SIZE * ratio))


def _make_loss(l6: float):
    def loss_fn(params, apply_fn, gas_states_log):
        pstar_nn = apply_fn({"params": params}, gas_states_log)
        gas_states_phys = physics.gas_log_to_phys(gas_states_log)
        pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
        mse = jnp.mean((pstar_nn - pstar_true) ** 2)
        l6_term = jnp.mean(jnp.abs((pstar_nn - pstar_true) ** 6))
        loss = (1.0 - l6) * mse + l6 * l6_term
        return loss, {"loss": loss}
    return loss_fn


def _make(name: str, train_domain: dict | None, l6: float) -> Experiment:
    assert 0.0 <= l6 <= 1.0
    bs = _BASE_BATCH_SIZE if train_domain is None else _scaled_batch(train_domain)
    return Experiment(
        name=name,
        model=PressureMLP(width=64, depth=2, normalize="geom"),
        domain=_DOMAIN,
        train_domain=train_domain,
        seed=42,
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        optax.cosine_decay_schedule(8e-3, 20_000, alpha=0.0),
                        weight_decay=1e-4,
                    ),
                ),
                n_epochs=20_000,
                loss=_make_loss(l6),
                batch_size=bs,
                log_every=200,
                name="adam_cosine",
            ),
        ],
    )


experiments = [
    _make("l6_0.0", _WIDE_TRAIN_DOMAIN, 0.0),
    _make("l6_0.1", _WIDE_TRAIN_DOMAIN, 0.1),
    _make("l6_0.4", _WIDE_TRAIN_DOMAIN, 0.4),
]
