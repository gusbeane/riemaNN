#!/usr/bin/env python3
"""Physics-informed NN for Riemann star pressure (ideal gas, gamma=5/3)."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0
ALPHA = (GAMMA - 1.0) / (2.0 * GAMMA)
BETA = (GAMMA - 1.0) / (GAMMA + 1.0)
MU = (GAMMA - 1.0) / 2.0


@jax.jit
def fjump(p, pK, rhoK):
    AK = 2.0 / ((GAMMA + 1.0) * rhoK)
    BK = pK * BETA
    cK = jnp.sqrt(GAMMA * pK / rhoK)
    shock = jnp.sqrt(AK / (p + BK)) * (p - pK)
    rarefaction = ((p / pK) ** ALPHA - 1.0) * cK / MU
    return jnp.where(p > pK, shock, rarefaction)


@jax.jit
def fstar(pstar, gas_state):
    rhoL, pL, rhoR, pR, uRL = gas_state
    return fjump(pstar, pL, rhoL) + fjump(pstar, pR, rhoR) + uRL


dfstar_dp = jax.grad(fstar, argnums=0)


@jax.jit
def find_pstar(gas_state, p0=1.0):
    def cond(state):
        pstar, fstar_, i = state
        return (jnp.abs(fstar_) >= 1e-6) & (i < 100)

    def body(state):
        pstar, _, i = state
        fstar_ = fstar(pstar, gas_state)
        dfstar_ = dfstar_dp(pstar, gas_state)
        pstar = pstar - fstar_ / dfstar_
        return pstar, fstar(pstar, gas_state), i + 1

    init = (p0, fstar(p0, gas_state), 0)
    pstar, fstar_final, _iters = jax.lax.while_loop(cond, body, init)
    return pstar, fstar_final


def gas_log_to_phys(gas_states_log):
    """(batch, 5): log10(rhoL,pL,rhoR,pR) + uRL -> physical (rhoL,pL,rhoR,pR,uRL)."""
    return jnp.concatenate([10.0 ** gas_states_log[:, :4], gas_states_log[:, 4:5]], axis=-1)


class StarPressureMLP(nn.Module):
    """Predicts log10(p*) from log-space left/right densities and pressures and uRL."""

    width: int = 64
    depth: int = 2  # number of hidden layers

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = nn.silu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


def create_train_state(
    rng,
    learning_rate,
    batch_size_hint: int = 256,
    width: int = 64,
    depth: int = 2,
):
    model = StarPressureMLP(width=width, depth=depth)
    dummy = jnp.ones((batch_size_hint, 5))
    params = model.init(rng, dummy)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, gas_states_log):
    def loss_fn(params):
        logpstar_nn = state.apply_fn({"params": params}, gas_states_log)
        pstar_nn = 10.0 ** logpstar_nn
        gas_states = gas_log_to_phys(gas_states_log)
        fstar_nn = jax.vmap(fstar)(pstar_nn, gas_states)
        return jnp.mean(fstar_nn**2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def ckpt_path(out_dir: Path, width: int, depth: int) -> Path:
    return out_dir / f"star_pressure_w{width}_d{depth}.msgpack"


def loss_npy_path(out_dir: Path, width: int, depth: int) -> Path:
    return out_dir / f"ideal_riemann_loss_w{width}_d{depth}.npy"


def sample_batch(rng, batch_size):
    keys = jr.split(rng, 6)
    logrhoL = jr.uniform(keys[0], (batch_size,), minval=-2.0, maxval=2.0)
    logpL = jr.uniform(keys[1], (batch_size,), minval=-2.0, maxval=2.0)
    logrhoR = jr.uniform(keys[2], (batch_size,), minval=-2.0, maxval=2.0)
    logpR = jr.uniform(keys[3], (batch_size,), minval=-2.0, maxval=2.0)
    uRL = jr.uniform(keys[4], (batch_size,), minval=-1.0, maxval=1.0)
    rng = keys[5]
    gas_states_log = jnp.stack([logrhoL, logpL, logrhoR, logpR, uRL], axis=-1)
    return rng, gas_states_log


def save_checkpoint(path: Path, state: train_state.TrainState) -> None:
    path.write_bytes(to_bytes(state))


def load_checkpoint(
    path: Path,
    learning_rate: float,
    batch_size: int,
    width: int,
    depth: int,
) -> train_state.TrainState:
    template = create_train_state(
        jr.PRNGKey(0),
        learning_rate,
        batch_size_hint=batch_size,
        width=width,
        depth=depth,
    )
    return from_bytes(template, path.read_bytes())


COMPARE_WIDTHS = (64, 128, 256, 512)
COMPARE_DEPTH = 2


def evaluate_holdout(state, n_samples: int = 20_000):
    """Median and 95th percentile of |f(p*)| and error vs exact p* on random states."""
    rng = jr.PRNGKey(999)
    _, gas_states_log = sample_batch(rng, n_samples)
    gas_states_phys = gas_log_to_phys(gas_states_log)

    logpstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
    pstar_nn = 10.0 ** logpstar_nn
    pstar_true, _ = jax.vmap(find_pstar)(gas_states_phys)
    f_nn = jax.vmap(fstar)(pstar_nn, gas_states_phys)

    abs_f = jnp.abs(f_nn)
    med_f = float(jnp.median(abs_f))
    p95_f = float(jnp.percentile(abs_f, 95.0))

    # |log10 p_NN - log10 p_true| only where both pressures are positive (log is defined).
    pos = (pstar_true > 1e-30) & (pstar_nn > 1e-30)
    abs_dlogp = jnp.abs(jnp.log10(pstar_nn) - jnp.log10(pstar_true))
    abs_dlogp = jnp.where(pos, abs_dlogp, jnp.nan)
    abs_dlogp_np = np.asarray(abs_dlogp)
    med_dlogp = float(np.nanmedian(abs_dlogp_np))
    p95_dlogp = float(np.nanpercentile(abs_dlogp_np, 95.0))
    frac_pos = float(jnp.mean(pos))

    # Robust relative pressure error (all samples): |p_NN - p_true| / max(|p_true|, eps).
    rel_p = jnp.abs(pstar_nn - pstar_true) / jnp.maximum(jnp.abs(pstar_true), 1e-30)
    med_rel = float(jnp.median(rel_p))
    p95_rel = float(jnp.percentile(rel_p, 95.0))

    return {
        "median_abs_f": med_f,
        "p95_abs_f": p95_f,
        "median_abs_delta_log10_p": med_dlogp,
        "p95_abs_delta_log10_p": p95_dlogp,
        "frac_both_p_positive": frac_pos,
        "median_rel_abs_p_err": med_rel,
        "p95_rel_abs_p_err": p95_rel,
    }


def train_or_load(
    out_dir: Path,
    width: int,
    depth: int,
    retrain: bool,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Returns (state, loss_trace or None if neither trained nor .npy on disk)."""
    ckpt = ckpt_path(out_dir, width, depth)
    loss_np = loss_npy_path(out_dir, width, depth)

    rng = jr.fold_in(jr.PRNGKey(42), width)
    rng, init_rng = jr.split(rng)

    if not retrain and ckpt.is_file():
        state = load_checkpoint(ckpt, learning_rate, batch_size, width, depth)
        loss_trace = None
        if loss_np.is_file():
            loss_trace = jnp.array(np.load(loss_np))
        return state, loss_trace

    state = create_train_state(
        init_rng,
        learning_rate,
        batch_size_hint=batch_size,
        width=width,
        depth=depth,
    )
    loss_trace = []

    pbar = tqdm(range(n_epochs), desc=f"w={width} d={depth}")
    for epoch in pbar:
        rng, gas_states_log = sample_batch(rng, batch_size)
        state, loss = train_step(state, gas_states_log)
        loss_trace.append(float(loss))
        if epoch % 2000 == 0:
            pbar.set_postfix(loss=f"{loss:.2e}")

    loss_trace = jnp.array(loss_trace)
    save_checkpoint(ckpt, state)
    np.save(loss_np, np.asarray(loss_trace))
    return state, loss_trace


def slice_grid_data():
    """Fixed slice in (log rho_L, log p_L) for comparison plots."""
    logrhoR_fix = 0.0
    logpR_fix = 0.0
    uRL_fix = 0.0
    N = 100
    logrhoL_vals = jnp.linspace(-2, 2, N)
    logpL_vals = jnp.linspace(-2, 2, N)
    logrhoL_grid, logpL_grid = jnp.meshgrid(logrhoL_vals, logpL_vals, indexing="ij")
    gas_states_log = jnp.stack(
        [
            logrhoL_grid.ravel(),
            logpL_grid.ravel(),
            jnp.full(N * N, logrhoR_fix),
            jnp.full(N * N, logpR_fix),
            jnp.full(N * N, uRL_fix),
        ],
        axis=-1,
    )
    return gas_states_log, logrhoL_vals, logpL_vals, N


def compute_slice_z(state, gas_states_log, N):
    logpstar_nn = state.apply_fn({"params": state.params}, gas_states_log)
    pstar_nn = 10.0 ** logpstar_nn
    gas_states_phys = gas_log_to_phys(gas_states_log)
    pstar_true, _ = jax.vmap(find_pstar)(gas_states_phys)
    f_nn = jax.vmap(fstar)(pstar_nn, gas_states_phys)
    log_ratio = jnp.log10(pstar_nn / pstar_true).reshape(N, N)
    signed_log_abs_f = (
        jnp.sign(f_nn) * jnp.log10(jnp.maximum(jnp.abs(f_nn), 1e-30))
    ).reshape(N, N)
    return np.asarray(log_ratio), np.asarray(signed_log_abs_f)


def plot_compare_loss(widths, depth, loss_traces, out_dir: Path) -> None:
    n = len(widths)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    y_lo, y_hi = None, None
    series = []
    for ax, w, lt in zip(axes, widths, loss_traces):
        if lt is not None:
            y = np.log10(np.asarray(lt))
            series.append(y)
            ax.plot(y, color="C0")
            y_lo = np.min(y) if y_lo is None else min(y_lo, np.min(y))
            y_hi = np.max(y) if y_hi is None else max(y_hi, np.max(y))
        ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
        ax.set_title(f"Training loss (width={w}, depth={depth})")
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
    path = out_dir / f"ideal_riemann_loss_compare_d{depth}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_compare_slice(widths, depth, states, out_dir: Path) -> None:
    gas_states_log, logrhoL_vals, logpL_vals, N = slice_grid_data()
    z1_list, z2_list = [], []
    for state in states:
        z1, z2 = compute_slice_z(state, gas_states_log, N)
        z1_list.append(z1)
        z2_list.append(z2)

    v1 = max(float(np.nanmax(np.abs(z))) for z in z1_list)
    v2 = max(float(np.nanmax(np.abs(z))) for z in z2_list)
    if not np.isfinite(v1) or v1 == 0.0:
        v1 = 1e-6
    if not np.isfinite(v2) or v2 == 0.0:
        v2 = 1e-6
    norm1 = TwoSlopeNorm(vmin=-v1, vcenter=0.0, vmax=v1)
    norm2 = TwoSlopeNorm(vmin=-v2, vcenter=0.0, vmax=v2)

    n = len(widths)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.8 * n), sharex=True, sharey=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, w in enumerate(widths):
        z1, z2 = z1_list[i], z2_list[i]
        ax1, ax2 = axes[i, 0], axes[i, 1]
        c1 = ax1.pcolormesh(
            logrhoL_vals,
            logpL_vals,
            z1.T,
            shading="auto",
            cmap="RdBu_r",
            norm=norm1,
        )
        ax1.set_ylabel(r"$\log_{10}p_L$")
        ax1.set_title(rf"$\log_{{10}}(p^*_{{\mathrm{{NN}}}}/p^*_{{\mathrm{{true}}}})$ — width={w}")
        fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)

        c2 = ax2.pcolormesh(
            logrhoL_vals,
            logpL_vals,
            z2.T,
            shading="auto",
            cmap="RdBu_r",
            norm=norm2,
        )
        ax2.set_title(rf"$\mathrm{{sign}}(f)\,\log_{{10}}|f(p^*_{{\mathrm{{NN}}}})|$ — width={w}")
        fig.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}\rho_L$")
    plt.tight_layout()
    path = out_dir / f"ideal_riemann_slice_compare_d{depth}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main_compare(
    retrain: bool = False,
    widths: tuple[int, ...] = COMPARE_WIDTHS,
    depth: int = COMPARE_DEPTH,
) -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    n_epochs = 5_000
    batch_size = 256
    learning_rate = 1e-3

    states = []
    loss_traces = []
    for w in widths:
        state, lt = train_or_load(
            out_dir, w, depth, retrain, n_epochs, batch_size, learning_rate
        )
        states.append(state)
        loss_traces.append(lt)
        metrics = evaluate_holdout(state)
        print(
            f"[width={w}, depth={depth}]  "
            f"|f|: median={metrics['median_abs_f']:.3e}, p95={metrics['p95_abs_f']:.3e}  |  "
            f"|Δlog10 p| (both p>0, frac={metrics['frac_both_p_positive']:.2f}): "
            f"median={metrics['median_abs_delta_log10_p']:.3e}, p95={metrics['p95_abs_delta_log10_p']:.3e}  |  "
            f"|p_NN-p_true|/|p_true|: median={metrics['median_rel_abs_p_err']:.3e}, "
            f"p95={metrics['p95_rel_abs_p_err']:.3e}"
        )

    if any(lt is not None for lt in loss_traces):
        plot_compare_loss(widths, depth, loss_traces, out_dir)
    plot_compare_slice(widths, depth, states, out_dir)


def main(retrain: bool = False, width: int = 64, depth: int = 2) -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    n_epochs = 5_000
    batch_size = 256
    learning_rate = 1e-3

    state, loss_trace = train_or_load(
        out_dir, width, depth, retrain, n_epochs, batch_size, learning_rate
    )

    metrics = evaluate_holdout(state)
    print(
        f"[width={width}, depth={depth}]  "
        f"|f|: median={metrics['median_abs_f']:.3e}, p95={metrics['p95_abs_f']:.3e}  |  "
        f"|Δlog10 p| (both p>0, frac={metrics['frac_both_p_positive']:.2f}): "
        f"median={metrics['median_abs_delta_log10_p']:.3e}, p95={metrics['p95_abs_delta_log10_p']:.3e}  |  "
        f"|p_NN-p_true|/|p_true|: median={metrics['median_rel_abs_p_err']:.3e}, "
        f"p95={metrics['p95_rel_abs_p_err']:.3e}"
    )

    if loss_trace is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(jnp.log10(loss_trace))
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
        ax.set_title(f"Training loss (w={width}, d={depth})")
        plt.tight_layout()
        fig.savefig(
            out_dir / f"ideal_riemann_loss_w{width}_d{depth}.png",
            dpi=150,
            bbox_inches="tight",
        )
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)

    gas_states_log, logrhoL_vals, logpL_vals, N = slice_grid_data()
    z1, z2 = compute_slice_z(state, gas_states_log, N)
    v1 = float(np.nanmax(np.abs(z1)))
    v2 = float(np.nanmax(np.abs(z2)))
    if not np.isfinite(v1) or v1 == 0.0:
        v1 = 1e-6
    if not np.isfinite(v2) or v2 == 0.0:
        v2 = 1e-6
    norm1 = TwoSlopeNorm(vmin=-v1, vcenter=0.0, vmax=v1)
    norm2 = TwoSlopeNorm(vmin=-v2, vcenter=0.0, vmax=v2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    c1 = ax1.pcolormesh(
        logrhoL_vals,
        logpL_vals,
        z1.T,
        shading="auto",
        cmap="RdBu_r",
        norm=norm1,
    )
    ax1.set_xlabel(r"$\log_{10}\rho_L$")
    ax1.set_ylabel(r"$\log_{10}p_L$")
    ax1.set_title(r"$\log_{10}(p^*_{\mathrm{NN}}/p^*_{\mathrm{true}})$")
    fig.colorbar(c1, ax=ax1)

    c2 = ax2.pcolormesh(
        logrhoL_vals,
        logpL_vals,
        z2.T,
        shading="auto",
        cmap="RdBu_r",
        norm=norm2,
    )
    ax2.set_xlabel(r"$\log_{10}\rho_L$")
    ax2.set_ylabel(r"$\log_{10}p_L$")
    ax2.set_title(r"$\mathrm{sign}(f)\,\log_{10}|f(p^*_{\mathrm{NN}})|$")
    fig.colorbar(c2, ax=ax2)

    plt.tight_layout()
    fig.savefig(
        out_dir / f"ideal_riemann_slice_w{width}_d{depth}.png",
        dpi=150,
        bbox_inches="tight",
    )
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load PINN for Riemann star pressure.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore saved checkpoint and train from scratch (overwrites checkpoint and loss .npy).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Train or load several widths (see --widths), print metrics, and save comparison "
            "loss/slice figures with shared y-axis and color limits."
        ),
    )
    parser.add_argument(
        "--widths",
        type=str,
        default="64,128,256,512",
        help="Comma-separated hidden widths for --compare (default: 64,128,256,512).",
    )
    parser.add_argument("--width", type=int, default=64, help="Hidden layer width (single run).")
    parser.add_argument("--depth", type=int, default=2, help="Number of hidden layers.")
    args = parser.parse_args()
    if args.compare:
        widths = tuple(int(x.strip()) for x in args.widths.split(",") if x.strip())
        main_compare(retrain=args.retrain, widths=widths, depth=args.depth)
    else:
        main(retrain=args.retrain, width=args.width, depth=args.depth)
