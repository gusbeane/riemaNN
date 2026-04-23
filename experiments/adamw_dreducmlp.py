"""Effect of widening the training domain beyond the evaluation domain.

Narrow: train_domain == domain (training = evaluation region).
Wide:   train_domain extended by +/- 0.2 in log rho/log p and +/- 0.1 in
        uRL, so some training samples fall outside the test region.

The wide run scales its batch size by the 5-D volume ratio so per-batch
sample density is held constant. Evaluation/plot domain is the same in
both runs.
"""

from math import prod

import optax

from riemann_pinn.model import CompactDimensionPressureMLP
from riemann_pinn.train import Experiment, Phase, supervised_loss


_BASE_BATCH_SIZE = 8096

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


def _phases(batch_size: int):
    return [
        Phase(
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    optax.cosine_decay_schedule(8e-3, 5_000, alpha=0.0),
                    weight_decay=1e-4,
                ),
            ),
            n_epochs=5_000,
            loss=supervised_loss,
            batch_size=batch_size,
            log_every=200,
            name="adam_cosine",
        ),
    ]


def _make(name: str, train_domain: dict | None) -> Experiment:
    bs = _BASE_BATCH_SIZE if train_domain is None else _scaled_batch(train_domain)
    return Experiment(
        name=name,
        model=CompactDimensionPressureMLP(width=64, depth=2),
        domain=_DOMAIN,
        train_domain=train_domain,
        seed=42,
        phases=_phases(bs),
    )


experiments = [
    _make("narrow", None),
    _make("wide", _WIDE_TRAIN_DOMAIN),
]
