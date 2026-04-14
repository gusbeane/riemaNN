"""Loss curve and slice heatmap plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from . import physics  # noqa: E402


def plot_loss(loss_trace, out_path: Path, *, title: str = "Training loss") -> None:
    arr = np.asarray(loss_trace)
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(np.log10(arr))
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slice(state, out_path: Path, *, n: int = 100) -> None:
    """Two-panel heatmap: log10(p_NN/p_true) and signed log|f(p_NN)|."""
    # Build 2D grid over (log rhoL, log pL) with right state and uRL fixed
    lr = jnp.linspace(-2.0, 2.0, n)
    lp = jnp.linspace(-2.0, 2.0, n)
    lr_grid, lp_grid = jnp.meshgrid(lr, lp, indexing="ij")
    gas_states_log = jnp.stack([
        lr_grid.ravel(), lp_grid.ravel(),
        jnp.zeros(n * n), jnp.zeros(n * n), jnp.zeros(n * n),
    ], axis=-1)

    # Evaluate
    raw_out = state.apply_fn({"params": state.params}, gas_states_log)
    pstar_nn = 10.0 ** raw_out
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states_phys)

    log_ratio = np.asarray(jnp.log10(pstar_nn / pstar_true).reshape(n, n))
    signed_f = np.asarray(
        (jnp.sign(fstar_vals) * jnp.log10(jnp.maximum(jnp.abs(fstar_vals), 1e-30)))
        .reshape(n, n)
    )

    lr_np = np.asarray(lr)
    lp_np = np.asarray(lp)
    titles = {
        "log_ratio": r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        "signed_f": r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, z, key in zip(axes, [log_ratio, signed_f], ["log_ratio", "signed_f"]):
        v = float(np.nanmax(np.abs(z)))
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        c = ax.pcolormesh(lr_np, lp_np, z.T, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_xlabel(r"$\log_{10}\rho_L$")
        ax.set_ylabel(r"$\log_{10}p_L$")
        ax.set_title(titles[key])
        fig.colorbar(c, ax=ax)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
