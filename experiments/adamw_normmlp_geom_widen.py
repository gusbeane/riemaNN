"""Effect of widening the training domain beyond the evaluation domain.

Base config is the w64_d2_lr0.008 cell of adamw_normmlp_wdgrid.py with the
geometric-mean normalization (StarPressureMLPNormalizedGeom). Only the
sampling domain during training changes:

  narrow:  train_domain == domain  (training = evaluation region)
  wide:    train_domain extended by +/- 0.2 in log rho, log p
                                   and +/- 0.1 in uRL, so some training
                                   samples fall outside the test region.

The wide run scales its batch size by the 5-D volume ratio so sample
density per batch is held constant. The evaluation/plot domain is the
same in both runs.
"""

from math import prod

from riemann_pinn.experiment import Experiment, adam_cosine
from riemann_pinn.model import StarPressureMLPNormalizedGeom

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
    # 5-D box volume: log rho appears twice (L, R), same for log p.
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
        adam_cosine(
            n_epochs=40_000, lr=8e-3, alpha=0.0, batch_size=batch_size,
            loss="fstar", log_every=200,
        ),
    ]


def _make(name: str, train_domain: dict | None) -> Experiment:
    bs = _BASE_BATCH_SIZE if train_domain is None else _scaled_batch(train_domain)
    return Experiment(
        model=StarPressureMLPNormalizedGeom(width=64, depth=2),
        domain=_DOMAIN,
        train_domain=train_domain,
        seed=42,
        name=name,
        phases=_phases(bs),
    )


experiments = [
    _make("narrow", None),
    _make("wide", _WIDE_TRAIN_DOMAIN),
]
