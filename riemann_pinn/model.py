"""MLP for predicting p* from a log-space gas state.

The network is parameterized internally in log-space (it emits log10 p*),
but `__call__` returns the physical p* so callers don't need to undo the
log transform.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import physics

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

class StarPressureMLP(nn.Module):
    """MLP that maps a log-space gas state (B, 5) to a scalar p*."""

    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        model = _MLP(width=self.width, depth=self.depth, activation=self.activation,
                 output_dim=self.output_dim)

        x = model(x)

        if self.output_dim == 1:
            x = x.squeeze(-1)

        return 10.0 ** x


class StarPressureMLPNormalized(nn.Module):
    """MLP that normalizes inputs/targets by state-dependent reference scales.

    Input x is (B, 5) in mixed form:
      (log10 rhoL, log10 pL, log10 rhoR, log10 pR, uRL).

    The network consumes:
      (log10(rhoL/rho_ref), log10(pL/p_ref),
       log10(rhoR/rho_ref), log10(pR/p_ref), uRL/u_ref),
    predicts log10(p*/p_ref), and returns physical p*.
    """

    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        model = _MLP(
            width=self.width,
            depth=self.depth,
            activation=self.activation,
            output_dim=1,
        )

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

class SirenMLP(nn.Module):
    """SIREN: MLP with sine activations.

    Known to represent smooth functions to very low MSE.  First layer uses
    the Sitzmann-recommended uniform init U(-1/n_in, 1/n_in) and applies
    sin(w0 * x) with w0=30 to bias the network toward higher frequencies.
    """

    width: int = 256
    depth: int = 3
    w0: float = 30.0
    output_dim: int = 1  # scalar log10 p*

    @nn.compact
    def __call__(self, x):
        n_in = x.shape[-1]

        # SIREN-specific init for the first layer.
        def siren_init_first(key, shape, dtype=jnp.float32):
            import jax.random as jr
            return jr.uniform(key, shape, dtype=dtype,
                              minval=-1.0 / n_in, maxval=1.0 / n_in)

        def siren_init_hidden(key, shape, dtype=jnp.float32):
            import jax.random as jr
            fan_in = shape[0]
            bound = jnp.sqrt(6.0 / fan_in) / self.w0
            return jr.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

        # First layer with special init
        h = nn.Dense(self.width, kernel_init=siren_init_first)(x)
        h = jnp.sin(self.w0 * h)

        # Hidden layers
        for _ in range(self.depth - 1):
            h = nn.Dense(self.width, kernel_init=siren_init_hidden)(h)
            h = jnp.sin(self.w0 * h)

        out = nn.Dense(self.output_dim, kernel_init=siren_init_hidden)(h)
        if self.output_dim == 1:
            out = out.squeeze(-1)
        return 10.0 ** out


class FourierFeatureMLP(nn.Module):
    """MLP with random Fourier feature encoding on inputs.

    Encodes x via (sin(Bx), cos(Bx)) with B a fixed random matrix drawn once
    at init, then feeds the encoding through a plain MLP. Fourier features
    help MLPs represent high-frequency smooth functions (NERF-style).
    """

    width: int = 256
    depth: int = 3
    n_freq: int = 64
    freq_scale: float = 4.0  # std of random frequencies
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        # Frozen random projection B: (5, n_freq). We register it as a
        # non-trainable variable so it doesn't update.
        B = self.param(
            "fourier_B",
            lambda key, shape: freq_scale_init(key, shape, self.freq_scale),
            (x.shape[-1], self.n_freq),
        )
        B_frozen = jax.lax.stop_gradient(B)
        proj = x @ B_frozen  # (B, n_freq)
        features = jnp.concatenate([jnp.sin(2 * jnp.pi * proj),
                                    jnp.cos(2 * jnp.pi * proj)],
                                   axis=-1)
        # Also include the raw input so low-frequency info is preserved.
        features = jnp.concatenate([x, features], axis=-1)

        h = features
        for _ in range(self.depth):
            h = nn.Dense(self.width)(h)
            h = self.activation(h)
        return 10.0 ** nn.Dense(1)(h).squeeze(-1)


def freq_scale_init(key, shape, scale):
    import jax.random as jr
    return scale * jr.normal(key, shape)


class StarPressureDS(nn.Module):
    """Deep Set that predicts p* from a log-space gas state.

    Shared encoder phi maps each (log rho, log p) gas state to a latent
    vector. The sum z = phi(xL) + phi(xR) is permutation-invariant.
    Decoder rho maps (z, uRL^2) to log10 p*; the module returns the
    exponentiated p*.
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
        return 10.0 ** rho(z).squeeze(-1)       # (B,)

