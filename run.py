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


def load_experiment(path: Path, index: int | None) -> tuple[Experiment, bool]:
    """Load an experiment module.

    The module must define either a single ``experiment = Experiment(...)`` or
    a list ``experiments = [Experiment(...), ...]``. Returns ``(exp, is_list)``;
    for list-valued modules, each experiment is required to carry a ``name``
    and ``index`` selects which one to return.
    """
    spec = importlib.util.spec_from_file_location("_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    has_single = hasattr(module, "experiment")
    has_list = hasattr(module, "experiments")
    if has_single and has_list:
        raise AttributeError(
            f"{path}: define exactly one of `experiment` or `experiments`, not both"
        )
    if not has_single and not has_list:
        raise AttributeError(
            f"{path}: must define `experiment = Experiment(...)` or "
            f"`experiments = [Experiment(...), ...]`"
        )

    if has_single:
        if index is not None:
            raise ValueError(
                f"{path}: defines a single `experiment`; --index is not allowed"
            )
        exp = module.experiment
        if not isinstance(exp, Experiment):
            raise TypeError(
                f"{path}: `experiment` must be an Experiment instance, "
                f"got {type(exp).__name__}"
            )
        return exp, False

    exps = module.experiments
    if not isinstance(exps, (list, tuple)) or not exps:
        raise TypeError(
            f"{path}: `experiments` must be a non-empty list of Experiment instances"
        )
    for i, e in enumerate(exps):
        if not isinstance(e, Experiment):
            raise TypeError(
                f"{path}: experiments[{i}] must be an Experiment instance, "
                f"got {type(e).__name__}"
            )
        if not e.name:
            raise ValueError(
                f"{path}: experiments[{i}] must set `name=...` when using a list"
            )
    names = [e.name for e in exps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate names in `experiments`: {names}")

    if index is None:
        raise ValueError(
            f"{path}: defines `experiments` (list of {len(exps)}); --index is required. "
            f"Available: {list(enumerate(names))}"
        )
    if not 0 <= index < len(exps):
        raise IndexError(
            f"{path}: --index {index} out of range [0, {len(exps)})"
        )
    return exps[index], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="select experiments[N] when the file defines a list")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--plot-corner-trace", action="store_true")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exp, is_list = load_experiment(exp_path, args.index)
    stem = exp_path.stem
    if is_list:
        name = f"{stem}/{exp.name}"
        out_dir = Path("outputs") / stem / exp.name
    else:
        name = stem
        out_dir = Path("outputs") / stem
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
        frames_dir = plots_dir / "corner_frames"
        def corner_cb(s, step):
            plot_corner_error(
                s, frames_dir / f"corner_error_{step:07d}.png", **exp.domain,
            )
        cb = corner_cb if args.plot_corner_trace else None
        t0 = time.monotonic()
        state, loss_trace, _ = run_experiment(exp, corner_callback=cb)
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
