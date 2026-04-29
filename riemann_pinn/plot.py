"""Loss curve, 3D-slice heatmap, hist2d, and corner plots (3 pairs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from . import physics  # noqa: E402
from .data import UniformSampler  # noqa: E402


_VAR_LABELS = [r"$\Delta\rho$", r"$\Delta p$", r"$\Delta u$"]


def plot_loss(loss_trace, out_path: Path, *, title: str = "Training loss") -> None:
    arr = np.asarray(loss_trace)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(np.log10(arr))
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
    ax.set_ylim(-8, 4)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slice(
    predict, out_path: Path, *, n: int = 250,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9),
    du_slice: float = 0.0,
    err_range=(-0.1, 0.1), nbins: int = 100, name: str | None = None,
) -> None:
    """Three-panel slice over (drho, dp) at du = du_slice. `predict` maps
    a (B, 3) gas-state batch to a (B,) pstar prediction."""
    dr = jnp.linspace(*drho_range, n)
    dpv = jnp.linspace(*dp_range, n)
    dr_grid, dp_grid = jnp.meshgrid(dr, dpv, indexing="ij")
    gas_states = jnp.stack([
        dr_grid.ravel(), dp_grid.ravel(),
        jnp.full(n * n, du_slice),
    ], axis=-1)

    pstar_nn = predict(gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)
    fstar_vals = jax.vmap(physics.fstar)(pstar_nn, gas_states)

    log_ratio = np.asarray(jnp.log10(pstar_nn / pstar_true).reshape(n, n))
    signed_f = np.asarray(
        (jnp.sign(fstar_vals) * jnp.log10(jnp.maximum(jnp.abs(fstar_vals), 1e-30)))
        .reshape(n, n)
    )

    dr_np = np.asarray(dr)
    dp_np = np.asarray(dpv)
    titles = {
        "log_ratio": r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        "signed_f": r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, z, key in zip(axes[:2], [log_ratio, signed_f], ["log_ratio", "signed_f"]):
        v = float(np.nanmax(np.abs(z)))
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        c = ax.pcolormesh(dr_np, dp_np, z.T, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_xlabel(_VAR_LABELS[0])
        ax.set_ylabel(_VAR_LABELS[1])
        ax.set_title(titles[key] + f" @ $\\Delta u={du_slice:g}$")
        fig.colorbar(c, ax=ax)

    ax = axes[2]
    bins = np.linspace(err_range[0], err_range[1], nbins)
    ax.hist(log_ratio.ravel(), bins=bins, histtype="step", ec="k", density=True,
            label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$")
    ax.set(xlim=err_range, yscale="log")

    swap_sign = jnp.array([-1.0, -1.0, 1.0])
    gas_states_swap = gas_states * swap_sign
    pstar_nn_swap = predict(gas_states_swap)
    log_ratio_sym = np.asarray(jnp.log10(pstar_nn) - jnp.log10(pstar_nn_swap))
    ax.hist(log_ratio_sym, bins=bins, histtype="step", ec="b", density=True,
            label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{NN,swap}})$")
    ax.legend(fontsize=8)

    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- corner plots ------------------------------------------------------------


def _corner_panels(n, *, drho_range, dp_range, du_range):
    ranges = [drho_range, dp_range, du_range]
    mids = [0.5 * (r[0] + r[1]) for r in ranges]
    axes = [jnp.linspace(r[0], r[1], n) for r in ranges]

    panels = []
    for row in range(2):
        for col in range(row + 1):
            x_var, y_var = col, row + 1
            xg, yg = jnp.meshgrid(axes[x_var], axes[y_var], indexing="ij")
            state_cols = []
            for k in range(3):
                if k == x_var:
                    state_cols.append(xg.ravel())
                elif k == y_var:
                    state_cols.append(yg.ravel())
                else:
                    state_cols.append(jnp.full(n * n, mids[k]))
            panels.append((row, col, jnp.stack(state_cols, axis=-1)))
    return panels, [np.asarray(a) for a in axes]


def plot_corner_error(
    predict, out_path: Path, *, n: int = 50,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_NN / p*_true) over all 3 input-variable pairs."""
    panels, axes_np = _corner_panels(
        n, drho_range=drho_range, dp_range=dp_range, du_range=du_range,
    )
    all_gas = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_nn_all = predict(all_gas)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(all_gas)
    log_ratio_all = np.asarray(jnp.log10(pstar_nn_all / pstar_true_all))

    panel_data = []
    offset = 0
    for row, col, gas in panels:
        npts = gas.shape[0]
        z = log_ratio_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmax = max(float(np.nanmax(np.abs(d[2]))) for d in panel_data)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 1].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="RdBu_r", norm=norm,
        )
        if row < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(_VAR_LABELS[col])
        if col > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(_VAR_LABELS[row + 1])

    fig.colorbar(
        mappable, ax=axarr.ravel().tolist(),
        label=r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        shrink=0.6, pad=0.02,
    )
    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corner_pstar(
    out_path: Path, *, n: int = 50,
    drho_range=(-0.9, 0.9), dp_range=(-0.9, 0.9), du_range=(-3.0, 0.9),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_true) over all 3 input-variable pairs."""
    panels, axes_np = _corner_panels(
        n, drho_range=drho_range, dp_range=dp_range, du_range=du_range,
    )
    all_gas = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(all_gas)
    log_pstar_all = np.asarray(jnp.log10(jnp.maximum(pstar_true_all, 1e-30)))

    panel_data = []
    offset = 0
    for row, col, gas in panels:
        npts = gas.shape[0]
        z = log_pstar_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmin = min(float(np.nanmin(d[2])) for d in panel_data)
    vmax = max(float(np.nanmax(d[2])) for d in panel_data)
    if not np.isfinite(vmin):
        vmin = -2.0
    if not np.isfinite(vmax):
        vmax = 2.0

    fig, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 1].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        if row < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(_VAR_LABELS[col])
        if col > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(_VAR_LABELS[row + 1])

    fig.colorbar(
        mappable, ax=axarr.ravel().tolist(),
        label=r"$\log_{10}\,p^*_{\mathrm{true}}$",
        shrink=0.6, pad=0.02,
    )
    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pstar_hist2d(
    predict, out_path: Path, *, n_samples: int = 50_000, seed: int = 999,
    nbins: int = 200, name: str | None = None,
    **domain_kwargs,
) -> None:
    """2D histogram of log10(p*_true) vs log10(p*_NN)."""
    import jax.random as jr

    rng = jr.PRNGKey(seed)
    sampler = UniformSampler(**domain_kwargs)
    gas_states = sampler.draw_batch(rng, n_samples)

    pstar_nn = predict(gas_states)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states)

    log_true = np.asarray(jnp.log10(jnp.maximum(pstar_true, 1e-30)))
    log_nn = np.asarray(jnp.log10(jnp.maximum(pstar_nn, 1e-30)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    lims = [min(log_true.min(), log_nn.min()), max(log_true.max(), log_nn.max())]
    bins = np.linspace(lims[0], lims[1], nbins)
    h = ax.hist2d(log_true, log_nn, bins=bins, norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="count")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"$\log_{10}\,p^*_{\mathrm{true}}$")
    ax.set_ylabel(r"$\log_{10}\,p^*_{\mathrm{NN}}$")
    title = r"$p^*$ prediction vs truth"
    if name:
        title = f"{title} -- {name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
