"""Train a PINN for Riemann star pressure from a TOML config file."""

import argparse
import json
import shutil
import time
import tomllib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.random as jr
import optax

from riemann_pinn.model import StarPressureDS, StarPressureMLP
from riemann_pinn.train import (
    create_train_state, evaluate_holdout,
    load_checkpoint, load_loss_trace,
    make_train_step, residual_loss, run_training_loop, save_checkpoint,
    save_loss_trace, uniform_log,
)
from riemann_pinn.plot import plot_loss, plot_pstar_hist2d, plot_slice


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_model(cfg):
    arch = cfg["model"]["arch"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    if arch == "mlp":
        return StarPressureMLP(width=width, depth=depth)
    elif arch == "ds":
        phi_output_dim = cfg["model"]["ds"]["phi_output_dim"]
        return StarPressureDS(
            phi_width=width, phi_depth=depth, phi_output_dim=phi_output_dim,
            rho_width=width, rho_depth=depth,
        )
    else:
        raise ValueError(f"Unknown model arch: {arch!r}")


def run_name(config_path: Path) -> str:
    return config_path.stem


def train_model(cfg, model, name):
    t = cfg["training"]
    domain = cfg["domain"]
    rng = jr.PRNGKey(t["seed"])
    schedule = optax.cosine_decay_schedule(init_value=t["lr"], decay_steps=t["n_epochs"])
    optimizer = optax.adam(learning_rate=schedule)
    state = create_train_state(rng, model, optimizer, batch_size_hint=t["batch_size"])
    step = make_train_step(residual_loss)

    domain_kw = dict(
        log_rho_range=tuple(domain["log_rho_range"]),
        log_p_range=tuple(domain["log_p_range"]),
        u_range=tuple(domain["u_range"]),
    )

    def sampler(rng, batch_size):
        return uniform_log(rng, batch_size, **domain_kw)

    state, trace = run_training_loop(
        state, step, sampler, jr.fold_in(rng, 1),
        n_epochs=t["n_epochs"], batch_size=t["batch_size"],
        desc=name, log_every=t["log_every"],
    )
    return state, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml", help="path to TOML config")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    t = cfg["training"]
    domain = cfg["domain"]
    domain_kw = dict(
        log_rho_range=tuple(domain["log_rho_range"]),
        log_p_range=tuple(domain["log_p_range"]),
        u_range=tuple(domain["u_range"]),
    )

    model = build_model(cfg)
    name = run_name(Path(args.config))

    out_dir = Path("outputs") / name
    ckpt_path = out_dir / "checkpoint.msgpack"
    loss_path = out_dir / "loss.npy"
    metrics_path = out_dir / "metrics.json"
    loss_plot = out_dir / "plots" / "loss.png"
    slice_plot = out_dir / "plots" / "slice.png"

    training_time_s = None
    if ckpt_path.is_file() and not args.retrain:
        print(f"Loading checkpoint from {ckpt_path}")
        rng = jr.PRNGKey(t["seed"])
        schedule = optax.cosine_decay_schedule(
            init_value=t["lr"], decay_steps=t["n_epochs"],
        )
        template = create_train_state(rng, model, optax.adam(learning_rate=schedule))
        state = load_checkpoint(ckpt_path, template)
        loss_trace = load_loss_trace(loss_path)
    else:
        t0 = time.monotonic()
        state, loss_trace = train_model(cfg, model, name)
        training_time_s = round(time.monotonic() - t0, 1)
        save_checkpoint(ckpt_path, state)
        save_loss_trace(loss_path, loss_trace)

    # Save config alongside outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.toml")

    # Evaluate
    metrics = evaluate_holdout(state, **domain_kw)
    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Plot
    if not args.skip_plots:
        if loss_trace is not None:
            plot_loss(loss_trace, loss_plot, title=f"Training loss — {name}")
        plot_slice(state, slice_plot,
                   log_rho_range=domain_kw["log_rho_range"],
                   log_p_range=domain_kw["log_p_range"])
        plot_pstar_hist2d(state, out_dir / "plots" / "pstar_hist2d.png",
                          **domain_kw)


if __name__ == "__main__":
    main()
