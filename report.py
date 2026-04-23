"""Print metrics tables for trained experiments.

Reads outputs/<file_stem>/<exp.name>/metrics.json for each experiment in
the given file and prints a compact table. Missing runs show 'nr'.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from run import load_experiments


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    def _out_dir(e):
        return (Path(e.output_root) if e.output_root else out_root / stem) / e.name

    loaded = [_load_metrics(_out_dir(e) / "metrics.json") for e in exps]

    if len(exps) == 1:
        m = loaded[0]
        if m is None:
            print(f"{stem}/{exps[0].name}: nr (no metrics.json in "
                  f"{_out_dir(exps[0])})")
            return
        width = max(len(_METRIC_LABELS.get(k, k)) for k in m) if m else 0
        for k in sorted(m):
            label = _METRIC_LABELS.get(k, k)
            print(f"{label:<{width}}  {_fmt_cell(m[k])}")
        return

    metric_keys: set[str] = set()
    for m in loaded:
        if m is not None:
            metric_keys.update(m.keys())
    # drop all-false boolean-like columns for readability
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


if __name__ == "__main__":
    main()
