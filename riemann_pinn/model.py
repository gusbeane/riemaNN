"""Neural network model that predicts dimensionless p* = p*/p_ref.

Input is a 3D delta-state (drho, dp, du). Output is log-scale to
guarantee positivity and to match the ~1-2 order-of-magnitude dynamic
range of p*/p_ref over the sampling domain.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


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
    """Maps (B, 3) delta state -> (B,) with configurable output transform."""
    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_transform: str | Callable[[jnp.ndarray], jnp.ndarray] = "pow10"

    def _apply_output_transform(self, y: jnp.ndarray) -> jnp.ndarray:
        if callable(self.output_transform):
            return self.output_transform(y)
        if self.output_transform == "pow10":
            return 10.0 ** y
        if self.output_transform == "tanh":
            return jnp.tanh(y)
        raise ValueError(
            "output_transform must be 'pow10', 'tanh', or a callable."
        )

    @nn.compact
    def __call__(self, x):  # x: (B, 3) = (drho, dp, du)
        model = _MLP(
            width=self.width, depth=self.depth,
            activation=self.activation, output_dim=1,
        )
        return self._apply_output_transform(model(x).squeeze(-1))
