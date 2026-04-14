"""MLP for predicting log10(p*) from a log-space gas state."""

from typing import Callable

import flax.linen as nn


class StarPressureMLP(nn.Module):
    """MLP that maps a log-space gas state (B, 5) to a scalar log10(p*)."""

    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x
