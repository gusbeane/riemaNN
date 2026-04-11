"""Generate comparison plots across a set of trained experiments.

Usage:
    venv/bin/python -m scripts.compare <name1> <name2> ... \\
        [--label-by <config-key>] [--out <outdir>]

For each experiment name, loads the saved state via Experiment.load(),
extracts a label (either the experiment name or the value of a config
key path like "model.width"), and writes two comparison plots
(loss.png and slice.png) to the output directory.

Example — reproduce the old width sweep:
    venv/bin/python -m scripts.compare \\
        baseline_w64_d2 baseline_w128_d2 baseline_w256_d2 baseline_w512_d2 \\
        --label-by model.width \\
        --out outputs/_compare/baselines_by_width
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

from riemann_pinn import Experiment, evaluation, paths, plotting, targets


def _dotted_lookup(d: dict, key: str) -> Any:
    """Follow a dotted key path like 'model.width' through nested dicts."""
    cur: Any = d
    for part in key.split("."):
        cur = cur[part]
    return cur


def _target_from_name(name: str):
    """Look up a Target by the name stored in config.json."""
    if name not in targets.TARGET_REGISTRY:
        raise KeyError(
            f"Target {name!r} not in TARGET_REGISTRY. "
            f"Known: {sorted(targets.TARGET_REGISTRY)}"
        )
    return targets.TARGET_REGISTRY[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="+", help="Experiment names to compare")
    ap.add_argument(
        "--label-by",
        default=None,
        help="Dotted config key for labels (e.g. 'model.width'). "
        "Default: the experiment name.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: outputs/_compare/<joined-names>",
    )
    args = ap.parse_args()

    out_dir = (
        Path(args.out)
        if args.out
        else paths.OUTPUTS_ROOT / "_compare" / "_".join(args.names)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [Experiment.load(name) for name in args.names]

    labels: list[str] = []
    for r in results:
        if args.label_by is None:
            labels.append(r.name)
        else:
            value = _dotted_lookup(r.config, args.label_by)
            labels.append(f"{args.label_by.split('.')[-1]}={value}")

    # Loss comparison.
    plotting.plot_compare_loss(
        labels,
        [r.loss_trace for r in results],
        out_dir / "loss.png",
    )

    # Slice comparison — build a shared slice grid and compute fields per experiment.
    gas_states_log, lrL, lpL, n = evaluation.slice_grid_data()
    slice_fields_list = []
    for r in results:
        target = _target_from_name(r.config["target"])
        slice_fields_list.append(
            evaluation.compute_slice_fields(r.state, target, gas_states_log, n)
        )
    plotting.plot_compare_slice(
        labels, slice_fields_list, lrL, lpL, out_dir / "slice.png"
    )

    # Print a summary table of metrics from each run.
    print(f"Comparison output: {out_dir}")
    for label, r in zip(labels, results):
        print(f"  {label}: {evaluation.format_metrics(r.metrics)}")


if __name__ == "__main__":
    main()
