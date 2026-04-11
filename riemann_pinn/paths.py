"""Path helpers for the outputs/<name>/... layout.

Single source of truth for where per-experiment artifacts live on disk.
If you want to reorganize the outputs directory later, this is the only
file that needs to change.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT: Path = REPO_ROOT / "outputs"


def experiment_dir(name: str) -> Path:
    d = OUTPUTS_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(name: str) -> Path:
    return experiment_dir(name) / "checkpoint.msgpack"


def loss_trace_path(name: str) -> Path:
    return experiment_dir(name) / "loss.npy"


def metrics_path(name: str) -> Path:
    return experiment_dir(name) / "metrics.json"


def config_snapshot_path(name: str) -> Path:
    return experiment_dir(name) / "config.json"


def plot_path(name: str, plot_name: str) -> Path:
    plots_dir = experiment_dir(name) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / f"{plot_name}.png"
