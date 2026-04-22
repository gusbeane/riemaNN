"""Neural network models that predict the star-region pressure p*.

A single `PressureMLP` with a `normalize` flag covers the three variants we
actually use (plain, arithmetic-mean-normalized, geometric-mean-normalized).
Output is physical p* so callers don't need to undo the log transform.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from . import physics


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
    """Maps log-space gas state (B, 5) -> scalar p*.

    normalize:
      "none"   — raw log inputs; network emits log10 p* directly.
      "arith"  — divide by arithmetic means: rho_ref = 0.5*(rhoL+rhoR),
                 p_ref = 0.5*(pL+pR), u_ref = c_s(p_ref, rho_ref).
      "geom"   — geometric means in log space (antisymmetric under L<->R).
    """
    width: int = 64
    depth: int = 2
    normalize: str = "none"
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        model = _MLP(width=self.width, depth=self.depth,
                     activation=self.activation, output_dim=1)

        if self.normalize == "none":
            return 10.0 ** model(x).squeeze(-1)

        if self.normalize == "arith":
            gas_phys = physics.gas_log_to_phys(x)
            rhoL, pL, rhoR, pR, uRL = jnp.split(gas_phys, [1, 2, 3, 4], axis=-1)
            rho_ref = 0.5 * (rhoL + rhoR)
            p_ref = 0.5 * (pL + pR)
            u_ref = physics.sound_speed(p_ref, rho_ref)
            gas_phys_norm = jnp.concatenate(
                [rhoL / rho_ref, pL / p_ref, rhoR / rho_ref, pR / p_ref, uRL / u_ref],
                axis=-1,
            )
            x_norm = physics.gas_phys_to_log(gas_phys_norm)
            log_pstar_over_pref = model(x_norm).squeeze(-1)
            return p_ref.squeeze(-1) * (10.0 ** log_pstar_over_pref)

        if self.normalize == "geom":
            log_rhoL, log_pL, log_rhoR, log_pR, uRL = jnp.split(x, 5, axis=-1)
            log_rho_ref = 0.5 * (log_rhoL + log_rhoR)
            log_p_ref = 0.5 * (log_pL + log_pR)
            p_ref = 10.0 ** log_p_ref
            rho_ref = 10.0 ** log_rho_ref
            u_ref = physics.sound_speed(p_ref, rho_ref)
            x_norm = jnp.concatenate(
                [
                    log_rhoL - log_rho_ref,
                    log_pL - log_p_ref,
                    log_rhoR - log_rho_ref,
                    log_pR - log_p_ref,
                    uRL / u_ref,
                ],
                axis=-1,
            )
            log_pstar_over_pref = model(x_norm).squeeze(-1)
            return p_ref.squeeze(-1) * (10.0 ** log_pstar_over_pref)

        raise ValueError(f"unknown normalize mode: {self.normalize!r}")
