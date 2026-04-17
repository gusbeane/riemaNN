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

from matplotlib.colors import LogNorm  # noqa: E402

from . import physics  # noqa: E402
from .train import uniform_log  # noqa: E402


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
    state, out_path: Path, *, n: int = 250,
    log_rho_range=(-1.0, 2.0), log_p_range=(-2.0, 2.0),
    err_range=(-0.1, 0.1), nbins=100, name: str | None = None,
) -> None:
    """Two-panel heatmap: log10(p_NN/p_true) and signed log|f(p_NN)|."""
    # Build 2D grid over (log rhoL, log pL) with right state and uRL fixed
    lr = jnp.linspace(*log_rho_range, n)
    lp = jnp.linspace(*log_p_range, n)
    lr_grid, lp_grid = jnp.meshgrid(lr, lp, indexing="ij")
    gas_states_log = jnp.stack([
        lr_grid.ravel(), lp_grid.ravel(),
        jnp.zeros(n * n), jnp.zeros(n * n), jnp.zeros(n * n),
    ], axis=-1)

    # Evaluate
    pstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, z, key in zip(axes[:2], [log_ratio, signed_f], ["log_ratio", "signed_f"]):
        v = float(np.nanmax(np.abs(z)))
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        c = ax.pcolormesh(lr_np, lp_np, z.T, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_xlabel(r"$\log_{10}\rho_L$")
        ax.set_ylabel(r"$\log_{10}p_L$")
        ax.set_title(titles[key])
        fig.colorbar(c, ax=ax)
    
    ax = axes[2]
    bins = np.linspace(err_range[0], err_range[1], nbins)
    ax.hist(log_ratio, bins=bins, histtype='step', ec='k', density=True)
    ax.set(xlim=err_range, yscale='log')

    # now do symmetry test
    gas_states_log_sym = gas_states_log[:, [2, 3, 0, 1, 4]]
    gas_states_log_sym = gas_states_log_sym.at[:, 4].multiply(-1)
    pstar_nn_sym = state.apply_fn({"params": state.params}, gas_states_log_sym)
    log_ratio_sym = jnp.log10(pstar_nn) - jnp.log10(pstar_nn_sym)

    ax.hist(log_ratio_sym, bins=bins, histtype='step', ec='b', density=True)

    if name:
        fig.suptitle(name)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Corner plots -------------------------------------------------------------

_VAR_LABELS = [
    r"$\log_{10}\rho_L$",
    r"$\log_{10}p_L$",
    r"$\log_{10}\rho_R$",
    r"$\log_{10}p_R$",
    r"$u_{RL}$",
]


def _corner_panels(n, *, log_rho_range, log_p_range, u_range):
    """Build gas-state grids for all 10 lower-triangle variable pairs.

    Returns ``(panels, axes_np)`` where *panels* is a list of
    ``(grid_row, grid_col, gas_states_log)`` and ``axes_np[k]`` is the
    1-D numpy array for variable *k*.
    """
    ranges = [log_rho_range, log_p_range, log_rho_range, log_p_range, u_range]
    mids = [0.5 * (r[0] + r[1]) for r in ranges]
    axes = [jnp.linspace(r[0], r[1], n) for r in ranges]

    panels = []
    for row in range(4):
        for col in range(row + 1):
            x_var, y_var = col, row + 1
            xg, yg = jnp.meshgrid(axes[x_var], axes[y_var], indexing="ij")
            state_cols = []
            for k in range(5):
                if k == x_var:
                    state_cols.append(xg.ravel())
                elif k == y_var:
                    state_cols.append(yg.ravel())
                else:
                    state_cols.append(jnp.full(n * n, mids[k]))
            panels.append((row, col, jnp.stack(state_cols, axis=-1)))
    return panels, [np.asarray(a) for a in axes]


def plot_corner_error(
    state, out_path: Path, *, n: int = 50,
    log_rho_range=(0.0, 2.0), log_p_range=(-2.0, 2.0), u_range=(-1.0, 1.0),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_NN / p*_true) over all input variable pairs."""
    panels, axes_np = _corner_panels(
        n, log_rho_range=log_rho_range, log_p_range=log_p_range, u_range=u_range,
    )

    all_gas_log = jnp.concatenate([g for _, _, g in panels], axis=0)
    pstar_nn_all = state.apply_fn({"params": state.params}, all_gas_log)
    gas_phys_all = physics.gas_log_to_phys(all_gas_log)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(gas_phys_all)
    log_ratio_all = np.asarray(jnp.log10(pstar_nn_all / pstar_true_all))

    panel_data = []
    offset = 0
    for row, col, gas_log in panels:
        npts = gas_log.shape[0]
        z = log_ratio_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmax = max(float(np.nanmax(np.abs(d[2]))) for d in panel_data)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))
    for r in range(4):
        for c in range(r + 1, 4):
            axarr[r, c].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="RdBu_r", norm=norm,
        )
        if row < 3:
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
    log_rho_range=(0.0, 2.0), log_p_range=(-2.0, 2.0), u_range=(-1.0, 1.0),
    name: str | None = None,
) -> None:
    """Corner plot of log10(p*_true) over all input variable pairs."""
    panels, axes_np = _corner_panels(
        n, log_rho_range=log_rho_range, log_p_range=log_p_range, u_range=u_range,
    )

    all_gas_log = jnp.concatenate([g for _, _, g in panels], axis=0)
    gas_phys_all = physics.gas_log_to_phys(all_gas_log)
    pstar_true_all, _ = jax.vmap(physics.find_pstar)(gas_phys_all)
    log_pstar_all = np.asarray(jnp.log10(jnp.maximum(pstar_true_all, 1e-30)))

    panel_data = []
    offset = 0
    for row, col, gas_log in panels:
        npts = gas_log.shape[0]
        z = log_pstar_all[offset:offset + npts].reshape(n, n)
        panel_data.append((row, col, z))
        offset += npts

    vmin = min(float(np.nanmin(d[2])) for d in panel_data)
    vmax = max(float(np.nanmax(d[2])) for d in panel_data)
    if not np.isfinite(vmin):
        vmin = -2.0
    if not np.isfinite(vmax):
        vmax = 2.0

    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))
    for r in range(4):
        for c in range(r + 1, 4):
            axarr[r, c].set_visible(False)

    mappable = None
    for row, col, z in panel_data:
        ax = axarr[row, col]
        x_var, y_var = col, row + 1
        mappable = ax.pcolormesh(
            axes_np[x_var], axes_np[y_var], z.T,
            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        if row < 3:
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
    state, out_path: Path, *, n_samples: int = 50_000, seed: int = 999,
    nbins : int = 200, name: str | None = None,
    **domain_kwargs,
) -> None:
    """2D histogram of log10(p*_true) vs log10(p*_NN)."""
    import jax.random as jr

    rng = jr.PRNGKey(seed)
    gas_states_log = uniform_log(rng, n_samples, **domain_kwargs)
    gas_states_phys = physics.gas_log_to_phys(gas_states_log)

    pstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
    pstar_true, _ = jax.vmap(physics.find_pstar)(gas_states_phys)

    log_true = np.asarray(jnp.log10(jnp.maximum(pstar_true, 1e-30)))
    log_nn = np.asarray(jnp.log10(jnp.maximum(pstar_nn, 1e-30)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    lims = [min(log_true.min(), log_nn.min()), max(log_true.max(), log_nn.max())]
    bins = np.linspace(lims[0], lims[1], nbins)
    h = ax.hist2d(log_true, log_nn, bins=bins, norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="count")

    # Shared axis range and perfect-prediction line
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"$\log_{10}\,p^*_{\mathrm{true}}$")
    ax.set_ylabel(r"$\log_{10}\,p^*_{\mathrm{NN}}$")
    title = r"$p^*$ prediction vs truth"
    if name:
        title = f"{title} — {name}"
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
