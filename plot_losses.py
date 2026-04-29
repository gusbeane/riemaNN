"""Overlay training-loss curves for every experiment in a file.

Reads outputs/<file_stem>/<exp.name>/<stage.name>/loss.npy for each stage
of each experiment, concatenates them in stage order, and writes a single
plot to outputs/<file_stem>/plots/loss_compare.png. Missing per-stage
loss traces are skipped with a warning."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import load_experiments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="path to Python experiment file")
    ap.add_argument("--out", default=None,
                    help="output path (default: outputs/<stem>/plots/loss_compare.png)")
    args = ap.parse_args()

    exp_path = Path(args.experiment)
    exps = load_experiments(exp_path)
    stem = exp_path.stem
    out_root = Path("outputs")

    def _exp_dir(e):
        parent = Path(e.output_root) if e.output_root else out_root / stem
        return parent / e.name

    traces: list[tuple[str, np.ndarray]] = []
    for e in exps:
        parts = []
        for s in e.stages:
            p = _exp_dir(e) / s.name / "loss.npy"
            if not p.is_file():
                print(f"warning: no loss.npy for {e.name}/{s.name} (expected {p}); skipping experiment")
                parts = None
                break
            parts.append(np.load(p))
        if parts:
            traces.append((e.name, np.concatenate(parts)))

    if not traces:
        raise SystemExit(f"no loss traces found for any experiment in {exp_path}")

    out_path = Path(args.out) if args.out else out_root / stem / "plots" / "loss_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for name, arr in traces:
        ax.plot(np.log10(arr), label=name, linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\log_{10}(\mathrm{loss})$")
    ax.set_ylim(-8, 4)
    ax.set_title(f"Training loss — {stem}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=max(1, len(traces) // 10))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(traces)} curves)")


if __name__ == "__main__":
    main()
