"""Checkpoint I/O: low-level model serialization plus the periodic-flush writer.

- `save_model` / `load_model` / `load_latest` are model-agnostic. A single
  `.msgpack` file holds `{"arch": <dict>, "params": <pure_dict>}`; loading
  rebuilds a skeleton via a caller-supplied `build_model` callback.
- `CheckpointWriter` orchestrates a training run's outputs: it buffers per-step
  losses, and on demand flushes them to `losses.txt`, evaluates the model via
  an injected `evaluate_fn`, appends a row to `metrics.csv`, and saves a
  versioned `F_net_{step:06d}.msgpack`. It depends only on `save_model`
  (above) and the two injected callables.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Mapping

import flax.nnx as nnx
import flax.serialization as fs
from tqdm import tqdm


def _stringify_int_keys(obj):
    """Recursively turn int dict-keys into strings.

    Needed because `nnx.List` produces integer indices in the pure-dict view
    of state, and `msgpack_restore` rejects integer map keys by default
    (`strict_map_key=True`).
    """
    if isinstance(obj, dict):
        return {(str(k) if isinstance(k, int) else k): _stringify_int_keys(v) for k, v in obj.items()}
    return obj


def _intify_digit_keys(obj):
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intify_digit_keys(v) for k, v in obj.items()}
    return obj


def save_model(model: nnx.Module, arch: dict, path: Path) -> None:
    """Save `model`'s params plus `arch` metadata to a single msgpack file.

    `arch` is opaque to this module — callers decide what they need to
    reconstruct a skeleton later. Its values must be msgpack-serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": dict(arch),
        "params": _stringify_int_keys(nnx.to_pure_dict(nnx.state(model))),
    }
    path.write_bytes(fs.msgpack_serialize(payload))


def load_model(path: Path, build_model: Callable[[dict], nnx.Module]) -> nnx.Module:
    """Load a model previously saved by `save_model`.

    `build_model(arch)` must return a fresh skeleton matching the saved
    architecture; its initial weights are immediately overwritten.
    """
    raw = fs.msgpack_restore(path.read_bytes())
    arch = raw["arch"]
    model = build_model(arch)
    state = nnx.state(model)
    nnx.replace_by_pure_dict(state, _intify_digit_keys(raw["params"]))
    nnx.update(model, state)
    return model


def load_latest(ckpt_dir: Path, build_model: Callable[[dict], nnx.Module]) -> nnx.Module:
    """Load the highest-numbered `F_net_*.msgpack` in `ckpt_dir`."""
    files = sorted(ckpt_dir.glob("F_net_*.msgpack"))
    if not files:
        raise FileNotFoundError(f"no F_net_*.msgpack in {ckpt_dir}")
    return load_model(files[-1], build_model)


# ---------------------------------------------------------------------------
# Periodic-flush writer

EvaluateFn = Callable[[nnx.Module], dict[str, float]]
PrintMetricsFn = Callable[[Mapping[str, float]], None]


class CheckpointWriter:
    """Owns all checkpoint I/O for a training run.

    `record_loss(loss_val)` is cheap and called every step.
    `flush(step, stage, F_net)` is called on every Nth step and at each stage's
    final step: appends pending losses to `losses.txt`, evaluates via
    `evaluate_fn`, appends a row to `metrics.csv`, and saves a versioned
    `F_net_{step:06d}.msgpack` via `save_model`.

    `evaluate_fn` and `print_metrics_fn` are injected so this class is
    independent of the physics / evaluation regimes.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        arch: dict,
        *,
        evaluate_fn: EvaluateFn,
        metric_keys: tuple[str, ...],
        print_metrics_fn: PrintMetricsFn | None = None,
    ):
        self.ckpt_dir = ckpt_dir
        self.arch = arch
        self.evaluate_fn = evaluate_fn
        self.metric_keys = metric_keys
        self.print_metrics_fn = print_metrics_fn

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for p in ckpt_dir.glob("F_net_*.msgpack"):
            p.unlink()
        self.losses_path = ckpt_dir / "losses.txt"
        self.metrics_path = ckpt_dir / "metrics.csv"
        self.losses_path.write_text("")
        with self.metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "stage", "loss", *metric_keys])

        self._pending: list = []

    def record_loss(self, loss_val) -> None:
        # `loss_val` may be a jax device array; defer the host sync until
        # `flush` so we don't block the training step.
        self._pending.append(loss_val)

    def flush(self, step: int, stage: str, F_net: nnx.Module) -> dict[str, float]:
        pending = [float(lv) for lv in self._pending]
        with self.losses_path.open("a") as f:
            f.writelines(f"{lv:.10e}\n" for lv in pending)
        last_loss = pending[-1] if pending else float("nan")
        self._pending.clear()

        metrics = self.evaluate_fn(F_net)
        with self.metrics_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [step, stage, f"{last_loss:.10e}", *(f"{metrics[k]:.10e}" for k in self.metric_keys)]
            )

        save_model(F_net, self.arch, self.ckpt_dir / f"F_net_{step:06d}.msgpack")

        tqdm.write(f"[flush] step={step} stage={stage} loss={last_loss:.4e}")
        if self.print_metrics_fn is not None:
            self.print_metrics_fn(metrics)
        return metrics
