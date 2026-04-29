"""Input pipeline: samplers that generate gas-state batches, and DataSet
for fixed precomputed pools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr


class Sampler(ABC):
    """Base class for samplers that generate gas-state batches on the fly."""

    def __init__(
        self,
        *,
        drho_range: tuple[float, float],
        dp_range: tuple[float, float],
        du_range: tuple[float, float],
    ):
        self.drho_range = drho_range
        self.dp_range = dp_range
        self.du_range = du_range

    @abstractmethod
    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        """Draw a batch of gas states. Returns (B, 3)."""
        ...


class UniformSampler(Sampler):
    """Uniform i.i.d. samples in (drho, dp, du)."""

    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        keys = jr.split(rng, 3)
        drho = jr.uniform(keys[0], (batch_size,), minval=self.drho_range[0], maxval=self.drho_range[1])
        dp   = jr.uniform(keys[1], (batch_size,), minval=self.dp_range[0],   maxval=self.dp_range[1])
        du   = jr.uniform(keys[2], (batch_size,), minval=self.du_range[0],   maxval=self.du_range[1])
        return jnp.stack([drho, dp, du], axis=-1)


# R2 quasirandom additive recurrence. Golden ratios for d=1..12.
_R2_GOLDEN = jnp.array([
    1.6180339887498949, 1.3247179572447463, 1.2207440846057596,
    1.1673039782614185, 1.1347241384015194, 1.1127756842787053,
    1.0969815577985598, 1.0850702454914507, 1.0757660660868371,
    1.0682971889208415, 1.0621691678642553, 1.0570505752212287,
])


class R2QuasirandomSampler(Sampler):
    """R2 quasirandom samples in (drho, dp, du)."""

    NDIM = 3

    def draw_batch(self, rng, batch_size: int) -> jnp.ndarray:
        g = _R2_GOLDEN[self.NDIM - 1]
        powers = jnp.arange(1, self.NDIM + 1, dtype=jnp.float32)
        a = g ** (-powers)
        x0 = jr.uniform(rng, (self.NDIM,), minval=0.0, maxval=1.0)
        n = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
        out_unit = jnp.mod(x0[None, :] + n * a[None, :], 1.0)
        lo = jnp.array([self.drho_range[0], self.dp_range[0], self.du_range[0]], dtype=out_unit.dtype)
        hi = jnp.array([self.drho_range[1], self.dp_range[1], self.du_range[1]], dtype=out_unit.dtype)
        return lo + (hi - lo) * out_unit


@dataclass
class DataSet:
    """Fixed precomputed pool of (gas_states, targets). The pipeline runner
    pulls gas_states from here but ignores `targets` -- targets are derived
    by the stage's make_targets from prev_running and pstar_true."""
    gas_states: jnp.ndarray   # (N, 3)
    targets:    jnp.ndarray | None  # (N,)
    head_idx:   int = 0

    def __post_init__(self):
        if self.targets is not None:
            assert self.gas_states.shape[0] == self.targets.shape[0], (
                "gas_states and targets must have the same number of samples"
            )

    def draw_batch(self, batch_size: int):
        assert self.head_idx + batch_size <= self.gas_states.shape[0], (
            "not enough samples left in the dataset"
        )
        start = self.head_idx
        end = self.head_idx + batch_size
        self.head_idx = end
        targets = None if self.targets is None else self.targets[start:end]
        return self.gas_states[start:end], targets
