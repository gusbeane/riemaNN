"""Run a PINN training experiment defined by a Python file.

Experiment files must export `experiments = [Experiment(...), ...]`.
Outputs land in outputs/<file_stem>/<exp.name>/<stage.name>/ for each
stage's checkpoint, loss, and per-stage metrics, plus a top-level
metrics.json and plots/ for the full pipeline.
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
    load_loss_trace, predict_pipeline, run_stage, save_checkpoint,
    save_loss_trace,
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
            raise ValueError(f"{path}: experiments[{i}] must set `name=...`")
        if not e.stages:
            raise ValueError(f"{path}: experiments[{i}] has empty `stages`")
        stage_names = [s.name for s in e.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError(
                f"{path}: experiments[{i}].stages has duplicate names: {stage_names}"
            )
        for j, s in enumerate(e.stages):
            if not s.name:
                raise ValueError(
                    f"{path}: experiments[{i}].stages[{j}] must set `name=...`"
                )
    names = [e.name for e in exps]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate names in `experiments`: {names}")
    return list(exps)


def _stage_dir(exp_dir: Path, stage_name: str) -> Path:
    return exp_dir / stage_name


def _wipe_from(exp_dir: Path, stages, retrain_from: str | None) -> None:
    """Delete the checkpoint dir for `retrain_from` and every later stage."""
    if retrain_from is None:
        return
    names = [s.name for s in stages]
    if retrain_from not in names:
        raise SystemExit(
            f"--retrain-from {retrain_from!r} not in this experiment's stages: {names}"
        )
    start = names.index(retrain_from)
    for s in stages[start:]:
        d = _stage_dir(exp_dir, s.name)
        if d.exists():
            print(f"  wiping {d}")
            shutil.rmtree(d)


def _train_pipeline(exp: Experiment, exp_dir: Path) -> tuple[list, list]:
    """Train each stage, skipping those with an existing checkpoint.

    Returns (stage_states, per_stage_traces) where stage_states is a list
    of (Stage, TrainState) and per_stage_traces is a list of np arrays
    aligned with exp.stages."""
    stage_states: list = []
    traces: list = []

    for i, stage in enumerate(exp.stages):
        sdir = _stage_dir(exp_dir, stage.name)
        ckpt_path = sdir / "checkpoint.msgpack"
        loss_path = sdir / "loss.npy"
        if ckpt_path.is_file():
            print(f"[{i+1}/{len(exp.stages)}] {stage.name}: loading checkpoint")
            state = load_checkpoint(ckpt_path, build_template_state(stage))
            trace = load_loss_trace(loss_path)
        else:
            print(f"[{i+1}/{len(exp.stages)}] {stage.name}: training")
            prev_specs = [
                (st.apply_fn, st.params, prev_stage.combine)
                for prev_stage, st in stage_states
            ]
            t0 = time.monotonic()
            state, trace, _ = run_stage(
                stage, prev_specs, exp_seed=exp.seed, stage_index=i,
            )
            elapsed = round(time.monotonic() - t0, 1)
            sdir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_path, state)
            save_loss_trace(loss_path, trace)
            (sdir / "stage_train_seconds.txt").write_text(f"{elapsed}\n")

        stage_states.append((stage, state))
        traces.append(trace)
    return stage_states, traces


def _train_and_eval(
    exp: Experiment, exp_path: Path, exp_dir: Path, name: str,
    *, retrain: bool, retrain_from: str | None, skip_plots: bool,
) -> None:
    if retrain and exp_dir.exists():
        print(f"  --retrain: wiping {exp_dir}")
        shutil.rmtree(exp_dir)
    _wipe_from(exp_dir, exp.stages, retrain_from)
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_path, exp_dir / exp_path.name)

    t0 = time.monotonic()
    stage_states, traces = _train_pipeline(exp, exp_dir)
    pipeline_time_s = round(time.monotonic() - t0, 1)

    # Per-stage metrics: evaluate the running pipeline truncated at each stage.
    for k, (stage, _state) in enumerate(stage_states):
        sdir = _stage_dir(exp_dir, stage.name)
        m = evaluate_holdout(stage_states[: k + 1], **exp.domain)
        with (sdir / "metrics.json").open("w") as f:
            json.dump(m, f, indent=2, sort_keys=True)
        if not skip_plots and traces[k] is not None:
            plot_loss(traces[k], sdir / "plots" / "loss.png",
                      title=f"Training loss — {name}/{stage.name}")

    # Whole-pipeline metrics.
    metrics = evaluate_holdout(stage_states, **exp.domain)
    metrics["training_time_s"] = pipeline_time_s
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if skip_plots:
        return
    predict = lambda gs: predict_pipeline(stage_states, gs)
    plots_dir = exp_dir / "plots"
    plot_slice(predict, plots_dir / "slice.png",
               drho_range=exp.domain["drho_range"],
               dp_range=exp.domain["dp_range"], name=name)
    plot_pstar_hist2d(predict, plots_dir / "pstar_hist2d.png", name=name, **exp.domain)
    plot_corner_error(predict, plots_dir / "corner_error.png", name=name, **exp.domain)
    plot_corner_pstar(plots_dir / "corner_pstar.png", name=name, **exp.domain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--index", type=int, default=None,
                    help="train only experiments[N] (default: all)")
    ap.add_argument("--retrain", action="store_true",
                    help="wipe and retrain every stage")
    ap.add_argument("--retrain-from", default=None,
                    help="wipe this stage and all later stages, then retrain")
    ap.add_argument("--skip-plots", action="store_true")
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
        exp_dir = parent / exp.name
        name = f"{stem}/{exp.name}"
        _train_and_eval(
            exp, exp_path, exp_dir, name,
            retrain=args.retrain, retrain_from=args.retrain_from,
            skip_plots=args.skip_plots,
        )


if __name__ == "__main__":
    main()
