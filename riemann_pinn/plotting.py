"""Plot functions for single-run and multi-run comparisons.

All functions take an explicit `out_path: Path` — the caller is
responsible for resolving where files go (typically via `paths.plot_path`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def _maybe_show(fig) -> None:
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_loss(
    loss_trace,
    out_path: Path,
    *,
    title: str = "Training loss",
    log10: bool = True,
) -> None:
    """Single-run loss curve."""
    arr = np.asarray(loss_trace)
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    y = np.log10(arr) if log10 else arr
    ax.plot(y)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$" if log10 else "loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _maybe_show(fig)


def plot_slice(
    slice_fields: dict,
    logrhoL_vals,
    logpL_vals,
    out_path: Path,
    *,
    titles: dict | None = None,
) -> None:
    """Two-panel heatmap for a single experiment: log_ratio and signed_log_abs_fstar."""
    keys = [k for k in ("log_ratio", "signed_log_abs_fstar") if k in slice_fields]
    if not keys:
        return
    default_titles = {
        "log_ratio": r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        "signed_log_abs_fstar": r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$",
    }
    titles = {**default_titles, **(titles or {})}

    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        z = slice_fields[key]
        v = float(np.nanmax(np.abs(z)))
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        c = ax.pcolormesh(
            np.asarray(logrhoL_vals),
            np.asarray(logpL_vals),
            z.T,
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )
        ax.set_xlabel(r"$\log_{10}\rho_L$")
        ax.set_ylabel(r"$\log_{10}p_L$")
        ax.set_title(titles[key])
        fig.colorbar(c, ax=ax)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _maybe_show(fig)


def plot_compare_loss(
    labels: list[str],
    loss_traces: list,
    out_path: Path,
    *,
    title_prefix: str = "Training loss",
) -> None:
    """Stacked subplots of multiple loss curves with a shared y-range."""
    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    y_lo = y_hi = None
    series = []
    for ax, label, lt in zip(axes, labels, loss_traces):
        if lt is not None:
            y = np.log10(np.asarray(lt))
            series.append(y)
            ax.plot(y, color="C0")
            y_lo = np.min(y) if y_lo is None else min(y_lo, np.min(y))
            y_hi = np.max(y) if y_hi is None else max(y_hi, np.max(y))
        ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
        ax.set_title(f"{title_prefix} — {label}")
        ax.grid(True, alpha=0.3)
    if y_lo is not None and y_hi is not None:
        pad = 0.05 * (y_hi - y_lo + 1e-15)
        for ax in axes:
            ax.set_ylim(y_lo - pad, y_hi + pad)
    if series:
        x_max = max(len(s) - 1 for s in series)
        for ax in axes:
            ax.set_xlim(0, x_max)
    axes[-1].set_xlabel("epoch")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _maybe_show(fig)


def plot_compare_slice(
    labels: list[str],
    slice_fields_list: list[dict],
    logrhoL_vals,
    logpL_vals,
    out_path: Path,
) -> None:
    """Grid of (runs × {log_ratio, signed_log_abs_fstar}) heatmaps with shared color limits."""
    keys = [k for k in ("log_ratio", "signed_log_abs_fstar") if
            any(k in sf for sf in slice_fields_list)]
    if not keys:
        return
    default_titles = {
        "log_ratio": r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$",
        "signed_log_abs_fstar": r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$",
    }

    # Shared norms per column.
    norms: dict[str, TwoSlopeNorm] = {}
    for key in keys:
        vals = [sf[key] for sf in slice_fields_list if key in sf]
        v = max(float(np.nanmax(np.abs(z))) for z in vals)
        if not np.isfinite(v) or v == 0.0:
            v = 1e-6
        norms[key] = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    n = len(labels)
    ncols = len(keys)
    fig, axes = plt.subplots(
        n, ncols, figsize=(6 * ncols, 2.8 * n), sharex=True, sharey=True
    )
    if n == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif n == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, (label, sf) in enumerate(zip(labels, slice_fields_list)):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            if key not in sf:
                ax.set_visible(False)
                continue
            z = sf[key]
            c = ax.pcolormesh(
                np.asarray(logrhoL_vals),
                np.asarray(logpL_vals),
                z.T,
                shading="auto",
                cmap="RdBu_r",
                norm=norms[key],
            )
            if j == 0:
                ax.set_ylabel(r"$\log_{10}p_L$")
            ax.set_title(f"{default_titles[key]} — {label}")
            fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}\rho_L$")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _maybe_show(fig)
