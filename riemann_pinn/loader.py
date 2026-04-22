"""Load experiment modules by path and resolve their output directories.

Kept separate from `run.py` so `riemann_pinn.experiment` can load a referenced
primary experiment (via `PrimarySpec`) without importing the CLI entry point.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .experiment import Experiment


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


def experiment_out_dir(root: Path, stem: str, exp: Experiment, is_list: bool) -> Path:
    return root / stem / exp.name if is_list else root / stem
