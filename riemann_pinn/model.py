"""MLP for predicting log10(p*) from a log-space gas state."""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


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

class _MLP(nn.Module):
    """Small reusable MLP block."""
    width: int = 64
    depth: int = 2
    output_dim: int = 5
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
        return nn.Dense(self.output_dim)(x)


class StarPressureDS(nn.Module):
    """Deep Set that predicts log10(p*) from log-space gas state.

    Shared encoder phi maps each (log rho, log p) gas state to a latent
    vector. The sum z = phi(xL) + phi(xR) is permutation-invariant.
    Decoder rho maps (z, uRL^2) to log10(p*).
    """

    phi_width: int = 64
    phi_depth: int = 2
    phi_output_dim: int = 5
    activation: Callable = nn.silu
    rho_width: int = 64
    rho_depth: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (B, 5) = (log rhoL, log pL, log rhoR, log pR, uRL)
        xL = x[:, :2]                          # (B, 2)
        xR = x[:, 2:4]                         # (B, 2)
        uRL_sq = x[:, 4:5] ** 2                # (B, 1) — even function for symmetry

        # Shared encoder: same instance called twice -> shared weights
        phi = _MLP(width=self.phi_width, depth=self.phi_depth,
                   output_dim=self.phi_output_dim, activation=self.activation,
                   name="phi")
        z = phi(xL) + phi(xR)                  # (B, phi_output_dim)
        z = jnp.concatenate([z, uRL_sq], axis=-1)  # (B, phi_output_dim + 1)

        # Decoder
        rho = _MLP(width=self.rho_width, depth=self.rho_depth,
                   output_dim=1, activation=self.activation,
                   name="rho")
        return rho(z).squeeze(-1)               # (B,)

