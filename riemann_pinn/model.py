"""Neural network model that predicts dimensionless p* = p*/p_ref.

Input is a 3D delta-state (drho, dp, du). Output passes through a
configurable transform; the default is 10**y, giving positive output
matched to the ~1-2 order-of-magnitude range of p*/p_ref over the
sampling domain.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


def _pow10(y: jnp.ndarray) -> jnp.ndarray:
    return 10.0 ** y


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
    """Maps (B, 3) delta state -> (B,) with a configurable output transform."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_transform: Callable[[jnp.ndarray], jnp.ndarray] = _pow10

    @nn.compact
    def __call__(self, x):  # x: (B, 3) = (drho, dp, du)
        model = _MLP(
            width=self.width, depth=self.depth,
            activation=self.activation, output_dim=1,
        )
        return self.output_transform(model(x).squeeze(-1))
