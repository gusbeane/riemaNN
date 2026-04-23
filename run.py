"""Run a PINN training experiment defined by a Python file.

Experiment files must export `experiments = [Experiment(...), ...]`
(single-element lists are fine). Outputs land in
outputs/<file_stem>/<exp.name>/.
"""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn.plot import (
    plot_corner_error, plot_corner_pstar, plot_loss,
    plot_pstar_hist2d, plot_slice,
)
from riemann_pinn.train import (
    Experiment, build_template_state, evaluate_holdout, load_checkpoint,
    load_loss_trace, run_experiment, save_checkpoint, save_loss_trace,
)


def load_experiments(path: Path) -> list[Experiment]:
    """Load `experiments = [Experiment(...), ...]` from a Python file."""
    spec = importlib.util.spec_from_file_location("_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "experiments"):
        raise AttributeError(
            f"{path}: must define `experiments = [Experiment(...), ...]`"
        )
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
                f"{path}: experiments[{i}] must set `name=...`"
            )
    names = [e.name for e in exps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate names in `experiments`: {names}")
    return list(exps)


def _train_and_eval(
    exp: Experiment, exp_path: Path, out_dir: Path, name: str,
    *, retrain: bool, skip_plots: bool, plot_corner_trace: bool,
) -> None:
    ckpt_path = out_dir / "checkpoint.msgpack"
    loss_path = out_dir / "loss.npy"
    metrics_path = out_dir / "metrics.json"
    plots_dir = out_dir / "plots"

    training_time_s = None
    if ckpt_path.is_file() and not retrain:
        print(f"Loading checkpoint from {ckpt_path}")
        state = load_checkpoint(ckpt_path, build_template_state(exp))
        loss_trace = load_loss_trace(loss_path)
    else:
        frames_dir = plots_dir / "corner_frames"

        def corner_cb(s, step):
            plot_corner_error(
                s, frames_dir / f"corner_error_{step:07d}.png",
                name=f"{name} @ step {step}", **exp.domain,
            )

        cb = corner_cb if plot_corner_trace else None
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

    if skip_plots:
        return
    if loss_trace is not None:
        plot_loss(loss_trace, plots_dir / "loss.png", title=f"Training loss — {name}")
    plot_slice(
        state, plots_dir / "slice.png",
        drho_range=exp.domain["drho_range"],
        dp_range=exp.domain["dp_range"],
        name=name,
    )
    plot_pstar_hist2d(state, plots_dir / "pstar_hist2d.png", name=name, **exp.domain)
    plot_corner_error(state, plots_dir / "corner_error.png", name=name, **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", name=name, **exp.domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="train only experiments[N] (default: all)")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--plot-corner-trace", action="store_true")
    ap.add_argument("--count", action="store_true",
                    help="print len(experiments) and exit")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    if args.count:
        print(len(exps))
        return

    if args.index is not None:
        if not 0 <= args.index < len(exps):
            raise IndexError(
                f"{exp_path}: --index {args.index} out of range [0, {len(exps)})"
            )
        selected = [exps[args.index]]
    else:
        selected = exps

    for exp in selected:
        parent = Path(exp.output_root) if exp.output_root else out_root / stem
        out_dir = parent / exp.name
        name = f"{stem}/{exp.name}"
        _train_and_eval(
            exp, exp_path, out_dir, name,
            retrain=args.retrain, skip_plots=args.skip_plots,
            plot_corner_trace=args.plot_corner_trace,
        )


if __name__ == "__main__":
    main()
