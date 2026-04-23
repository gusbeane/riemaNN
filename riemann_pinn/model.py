"""Neural network model that predicts dimensionless p* = p*/p_ref.

Input is a 3D delta-state (drho, dp, du). Output is log-scale to
guarantee positivity and to match the ~1-2 order-of-magnitude dynamic
range of p*/p_ref over the sampling domain.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp  # noqa: F401 (kept for downstream experimenters)


class _MLP(nn.Module):
    """Shared MLP block."""
    width: int
    depth: int
    output_dim: int = 1
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = self.activation(nn.Dense(self.width)(x))
        return nn.Dense(self.output_dim)(x)


class PressureMLP(nn.Module):
    """Maps (B, 3) delta state -> (B,) p*/p_ref, positive by log output."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):  # x: (B, 3) = (drho, dp, du)
        model = _MLP(
            width=self.width, depth=self.depth,
            activation=self.activation, output_dim=1,
        )
        return 10.0 ** model(x).squeeze(-1)
