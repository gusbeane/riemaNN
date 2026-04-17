"""Run a PINN training experiment defined by a Python file."""

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

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


def load_experiments(path: Path) -> tuple[list[Experiment], bool]:
    """Load an experiment module.

    The module must define either a single ``experiment = Experiment(...)`` or
    a list ``experiments = [Experiment(...), ...]``. Returns
    ``(experiments, is_list)``: a single-valued module is normalized to a
    one-element list with ``is_list=False``. For list-valued modules, every
    entry must carry a unique ``name``.
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
        exp = module.experiment
        if not isinstance(exp, Experiment):
            raise TypeError(
                f"{path}: `experiment` must be an Experiment instance, "
                f"got {type(exp).__name__}"
            )
        return [exp], False

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
    return list(exps), True


def select_index(exps: list[Experiment], is_list: bool, index: int | None,
                 path: Path) -> int:
    """Resolve an --index request against a loaded list.

    Required for lists; forbidden for single-experiment files.
    """
    if not is_list:
        if index is not None:
            raise ValueError(
                f"{path}: defines a single `experiment`; --index is not allowed"
            )
        return 0
    if index is None:
        raise ValueError(
            f"{path}: defines `experiments` (list of {len(exps)}); --index is required. "
            f"Available: {list(enumerate(e.name for e in exps))}"
        )
    if not 0 <= index < len(exps):
        raise IndexError(
            f"{path}: --index {index} out of range [0, {len(exps)})"
        )
    return index


def _experiment_out_dir(root: Path, stem: str, exp: Experiment, is_list: bool) -> Path:
    return root / stem / exp.name if is_list else root / stem


# --- metrics printing ---------------------------------------------------------

# Compact column labels for metrics coming out of evaluate_holdout. Keys not
# present here fall back to their raw name.
_METRIC_LABELS: dict[str, str] = {
    "median_abs_fstar":          "med|f(p*)|",
    "p95_abs_fstar":             "p95|f(p*)|",
    "median_abs_delta_log10_p":  "med|dlog10p*|",
    "p95_abs_delta_log10_p":     "p95|dlog10p*|",
    "abs_absolute_median":       "med|dp*|",
    "abs_absolute_p5":           "p5|dp*|",
    "abs_absolute_p95":          "p95|dp*|",
    "any_nan_nn":                "nan_nn",
    "any_nan_true":              "nan_tr",
    "any_neg_nn":                "neg_nn",
    "any_neg_true":              "neg_tr",
    "training_time_s":           "t (s)",
}


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "nr"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def print_metrics(exps: list[Experiment], is_list: bool, stem: str,
                  out_root: Path) -> None:
    """Print metrics for the experiment(s). Missing runs show 'nr'."""
    loaded = [
        _load_metrics(_experiment_out_dir(out_root, stem, e, is_list) / "metrics.json")
        for e in exps
    ]

    if not is_list:
        m = loaded[0]
        if m is None:
            print(f"{stem}: nr (no metrics.json in {out_root / stem})")
            return
        width = max(len(_METRIC_LABELS.get(k, k)) for k in m) if m else 0
        for k in sorted(m):
            label = _METRIC_LABELS.get(k, k)
            print(f"{label:<{width}}  {_fmt_cell(m[k])}")
        return

    # Metric columns: union across loaded runs, sorted; drop all-false booleans
    metric_keys: set[str] = set()
    for m in loaded:
        if m is not None:
            metric_keys.update(m.keys())
    for k in list(metric_keys):
        vals = {m[k] for m in loaded if m is not None and k in m}
        if vals and vals.issubset({"false"}):
            metric_keys.discard(k)
    metric_cols = sorted(metric_keys)
    if "training_time_s" in metric_cols:
        metric_cols.remove("training_time_s")
        metric_cols.append("training_time_s")

    header = ["name"] + [_METRIC_LABELS.get(k, k) for k in metric_cols]
    rows: list[list[str]] = []
    for e, m in zip(exps, loaded):
        if m is None:
            row = [e.name] + ["nr"] * len(metric_cols)
        else:
            row = [e.name] + [_fmt_cell(m.get(k)) for k in metric_cols]
        rows.append(row)

    _print_table(rows, header)


def _train_and_eval(exp: Experiment, exp_path: Path, out_dir: Path, name: str,
                    *, retrain: bool, skip_plots: bool,
                    plot_corner_trace: bool) -> None:
    """Train or load a single experiment, save artifacts, and (optionally) plot."""
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
        log_rho_range=exp.domain["log_rho_range"],
        log_p_range=exp.domain["log_p_range"],
        name=name,
    )
    plot_pstar_hist2d(state, plots_dir / "pstar_hist2d.png", name=name, **exp.domain)
    plot_corner_error(state, plots_dir / "corner_error.png", name=name, **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", name=name, **exp.domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="select experiments[N] when the file defines a list")
    ap.add_argument("--retrain", action="store_true", help="ignore checkpoint")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--plot-corner-trace", action="store_true")
    ap.add_argument("--print-metrics", action="store_true",
                    help="print metrics (aggregated over a list) and exit; "
                         "with --retrain, train the selected experiment first")
    ap.add_argument("--count", action="store_true",
                    help="print the number of experiments in the file and exit "
                         "(1 for single-experiment files)")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps, is_list = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    if args.count:
        print(len(exps))
        return

    # --print-metrics short-circuits: no plotting, optional retraining of one
    # selected experiment, then print metrics and exit.
    if args.print_metrics:
        if args.retrain:
            idx = select_index(exps, is_list, args.index, exp_path)
            exp = exps[idx]
            out_dir = _experiment_out_dir(out_root, stem, exp, is_list)
            name = f"{stem}/{exp.name}" if is_list else stem
            _train_and_eval(
                exp, exp_path, out_dir, name,
                retrain=True, skip_plots=True,
                plot_corner_trace=args.plot_corner_trace,
            )
        elif args.index is not None:
            raise ValueError(
                "--index is only meaningful with --retrain under --print-metrics; "
                "drop --index to see the aggregated table for all runs"
            )
        print_metrics(exps, is_list, stem, out_root)
        return

    idx = select_index(exps, is_list, args.index, exp_path)
    exp = exps[idx]
    out_dir = _experiment_out_dir(out_root, stem, exp, is_list)
    name = f"{stem}/{exp.name}" if is_list else stem
    _train_and_eval(
        exp, exp_path, out_dir, name,
        retrain=args.retrain, skip_plots=args.skip_plots,
        plot_corner_trace=args.plot_corner_trace,
    )


if __name__ == "__main__":
    main()
