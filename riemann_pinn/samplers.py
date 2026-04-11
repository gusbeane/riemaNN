"""Samplers for log-space gas-state batches.

Contract: every sampler has the signature `(rng, batch_size) -> (B, 5)
array of log-space gas states`. The training loop handles rng splitting,
so samplers do not need to return an updated key.

To parameterize a sampler (e.g. change the log-pressure range), use
`functools.partial` or `uniform_log_factory`.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.random as jr


def uniform_log(
    rng,
    batch_size: int,
    *,
    log_rho_range: tuple[float, float] = (-2.0, 2.0),
    log_p_range: tuple[float, float] = (-2.0, 2.0),
    u_range: tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Uniform samples in `(log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL)`."""
    keys = jr.split(rng, 5)
    log_rho_lo, log_rho_hi = log_rho_range
    log_p_lo, log_p_hi = log_p_range
    u_lo, u_hi = u_range
    logrhoL = jr.uniform(keys[0], (batch_size,), minval=log_rho_lo, maxval=log_rho_hi)
    logpL = jr.uniform(keys[1], (batch_size,), minval=log_p_lo, maxval=log_p_hi)
    logrhoR = jr.uniform(keys[2], (batch_size,), minval=log_rho_lo, maxval=log_rho_hi)
    logpR = jr.uniform(keys[3], (batch_size,), minval=log_p_lo, maxval=log_p_hi)
    uRL = jr.uniform(keys[4], (batch_size,), minval=u_lo, maxval=u_hi)
    return jnp.stack([logrhoL, logpL, logrhoR, logpR, uRL], axis=-1)


def uniform_log_factory(**kwargs) -> Callable:
    """Bind sampler kwargs up-front for ergonomic experiment configs.

    Example:
        sampler=uniform_log_factory(log_p_range=(-3, 3))
    """
    return partial(uniform_log, **kwargs)


# Future stubs — leave as comments for discoverability:
# - importance_log(rng, batch_size, weight_fn): adaptive sampling biased by residual
# - mixture(rng, batch_size, samplers, weights): curriculum / multi-regime blends
# - near_shock(rng, batch_size): biased toward |log pL - log pR| far from 0
