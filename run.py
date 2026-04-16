"""Run a PINN training experiment defined by a Python file."""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn.experiment import Experiment, build_template_state, run_experiment
from riemann_pinn.plot import (
    plot_corner_error,
    plot_corner_pstar,
    plot_loss,
    plot_pstar_hist2d,
    plot_slice,
)
from riemann_pinn.train import (
    evaluate_holdout,
    load_checkpoint,
    load_loss_trace,
    save_checkpoint,
    save_loss_trace,
)


def load_experiment(path: Path) -> Experiment:
    spec = importlib.util.spec_from_file_location("_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "experiment"):
        raise AttributeError(f"{path} must define `experiment = Experiment(...)`")
    exp = module.experiment
    if not isinstance(exp, Experiment):
        raise TypeError(
            f"{path}: `experiment` must be an Experiment instance, got {type(exp).__name__}"
        )
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exp = load_experiment(exp_path)
    name = exp_path.stem

    out_dir = Path("outputs") / name
    ckpt_path = out_dir / "checkpoint.msgpack"
    loss_path = out_dir / "loss.npy"
    metrics_path = out_dir / "metrics.json"
    plots_dir = out_dir / "plots"

    training_time_s = None
    if ckpt_path.is_file() and not args.retrain:
        print(f"Loading checkpoint from {ckpt_path}")
        state = load_checkpoint(ckpt_path, build_template_state(exp))
        loss_trace = load_loss_trace(loss_path)
    else:
        t0 = time.monotonic()
        state, loss_trace, _ = run_experiment(exp)
        training_time_s = round(time.monotonic() - t0, 1)
        save_checkpoint(ckpt_path, state)
        save_loss_trace(loss_path, loss_trace)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_path, out_dir / exp_path.name)

    metrics = evaluate_holdout(state, **exp.domain)
    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.skip_plots:
        return
    if loss_trace is not None:
        plot_loss(loss_trace, plots_dir / "loss.png", title=f"Training loss — {name}")
    plot_slice(
        state, plots_dir / "slice.png",
        log_rho_range=exp.domain["log_rho_range"],
        log_p_range=exp.domain["log_p_range"],
    )
    plot_pstar_hist2d(state, plots_dir / "pstar_hist2d.png", **exp.domain)
    plot_corner_error(state, plots_dir / "corner_error.png", **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", **exp.domain)


if __name__ == "__main__":
    main()
