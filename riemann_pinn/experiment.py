"""Experiment orchestration: ties together model, target, sampler, loss, training.

Each experiment script in `experiments/` instantiates one `Experiment` and
calls `.run()`. The class handles:

  - Checkpoint reuse (skip training if `outputs/<name>/checkpoint.msgpack`
    exists, unless `force_retrain=True`).
  - Config snapshot to `config.json` so `Experiment.load(name)` can
    reconstruct the model for the compare script.
  - Holdout evaluation + metrics.json.
  - Loss and slice plots to `plots/`.
  - `extra_evaluators` / `extra_plots` hooks so ad-hoc experiments can
    register additional metrics and figures without subclassing.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.random as jr
import numpy as np

from . import (
    evaluation,
    losses,
    models as models_module,
    paths,
    plotting,
    targets as targets_module,
    training,
)
from .targets import Target


@dataclass
class ExperimentResult:
    name: str
    config: dict[str, Any]
    state: Any
    loss_trace: np.ndarray | None
    metrics: dict[str, Any]


@dataclass
class Experiment:
    name: str
    model: nn.Module
    target: Target
    sampler: Callable
    loss_impl: losses.LossFn

    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(
        default_factory=lambda: {"type": "adam", "learning_rate": 1e-3}
    )
    n_epochs: int = 5_000
    batch_size: int = 256
    seed: int = 42
    eval_n_samples: int = 20_000

    extra_evaluators: dict[str, Callable] = field(default_factory=dict)
    extra_plots: list[Callable] = field(default_factory=list)

    # --- public API -----------------------------------------------------------

    def run(
        self, *, force_retrain: bool = False, skip_plots: bool = False
    ) -> ExperimentResult:
        """Train (if no checkpoint) or load, then evaluate and plot."""
        return self.run_custom(
            self._default_train_fn,
            force_retrain=force_retrain,
            skip_plots=skip_plots,
        )

    def run_custom(
        self,
        train_fn: Callable[["Experiment"], tuple[Any, Any]],
        *,
        force_retrain: bool = False,
        skip_plots: bool = False,
    ) -> ExperimentResult:
        """Like `run`, but the caller supplies the training function.

        `train_fn(exp) -> (state, loss_trace)` is responsible for building
        its own train state and running however many training phases it
        wants. The Experiment handles config snapshot, checkpoint reuse,
        save-on-success, evaluation, metrics, and plots.
        """
        paths.experiment_dir(self.name)
        config = self._resolve_config()
        self._write_config_snapshot(config)

        ckpt = paths.checkpoint_path(self.name)
        if ckpt.is_file() and not force_retrain:
            state, loss_trace = self._load_existing()
        else:
            state, loss_trace = train_fn(self)
            training.save_checkpoint(ckpt, state)
            if loss_trace is not None:
                training.save_loss_trace(
                    paths.loss_trace_path(self.name), loss_trace
                )

        metrics = self._evaluate(state)
        with paths.metrics_path(self.name).open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(evaluation.format_metrics(metrics, prefix=f"[{self.name}]"))

        if not skip_plots:
            self._make_plots(state, loss_trace)

        # Update the cross-experiment summary table.
        from scripts.summarize import write_summary

        try:
            write_summary()
        except Exception:
            pass  # non-critical; don't crash the experiment

        return ExperimentResult(
            name=self.name,
            config=config,
            state=state,
            loss_trace=None if loss_trace is None else np.asarray(loss_trace),
            metrics=metrics,
        )

    @classmethod
    def load(cls, name: str) -> ExperimentResult:
        """Rehydrate a past experiment from outputs/<name>/ without re-running."""
        config_path = paths.config_snapshot_path(name)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No config.json at {config_path}; run the experiment first."
            )
        with config_path.open() as f:
            config = json.load(f)

        model = models_module.build_model(config["model"])
        optimizer = training.build_optimizer(config["optimizer"])
        rng = jr.PRNGKey(config.get("seed", 0))
        template = training.create_train_state(
            rng, model, optimizer, batch_size_hint=config.get("batch_size", 256)
        )
        ckpt = paths.checkpoint_path(name)
        if not ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        state = training.load_checkpoint(ckpt, template)

        loss_trace = training.load_loss_trace(paths.loss_trace_path(name))

        metrics_file = paths.metrics_path(name)
        metrics = {}
        if metrics_file.is_file():
            with metrics_file.open() as f:
                metrics = json.load(f)

        return ExperimentResult(
            name=name,
            config=config,
            state=state,
            loss_trace=None if loss_trace is None else np.asarray(loss_trace),
            metrics=metrics,
        )

    def describe(self) -> str:
        return json.dumps(self._resolve_config(), indent=2, sort_keys=True)

    # --- private helpers ------------------------------------------------------

    def _resolve_config(self) -> dict[str, Any]:
        """Build the JSON-serializable config snapshot."""
        return {
            "name": self.name,
            "model": models_module.model_spec(self.model),
            "target": self.target.name,
            "loss_impl": getattr(self.loss_impl, "__name__", str(self.loss_impl)),
            "loss_kwargs": self.loss_kwargs,
            "optimizer": self.optimizer,
            "sampler": getattr(self.sampler, "__name__", str(self.sampler)),
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "eval_n_samples": self.eval_n_samples,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    def _write_config_snapshot(self, config: dict[str, Any]) -> None:
        with paths.config_snapshot_path(self.name).open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    def _build_template_state(self):
        optimizer = training.build_optimizer(self.optimizer)
        rng = jr.PRNGKey(self.seed)
        return training.create_train_state(
            rng, self.model, optimizer, batch_size_hint=self.batch_size
        )

    def _default_train_fn(self, exp: "Experiment"):
        """Default single-phase training function used by `run()`.

        `exp` is always `self`; the parameter exists so `_default_train_fn`
        matches the `(exp) -> (state, loss_trace)` signature of a custom
        train function and can be passed to `run_custom`.
        """
        state = exp._build_template_state()
        loss_fn = losses.make_loss_fn(
            exp.target, exp.loss_impl, **exp.loss_kwargs
        )
        train_step = training.make_train_step(loss_fn)
        training_rng = jr.fold_in(jr.PRNGKey(exp.seed), 1)
        state, loss_trace = training.run_training_loop(
            state,
            train_step,
            exp.sampler,
            training_rng,
            n_epochs=exp.n_epochs,
            batch_size=exp.batch_size,
            desc=exp.name,
        )
        return state, loss_trace

    def _load_existing(self):
        template = self._build_template_state()
        state = training.load_checkpoint(paths.checkpoint_path(self.name), template)
        loss_trace = training.load_loss_trace(paths.loss_trace_path(self.name))
        return state, loss_trace

    def _evaluate(self, state) -> dict[str, Any]:
        metrics = evaluation.evaluate_holdout(
            state, self.target, n_samples=self.eval_n_samples, seed=999
        )
        for key, evaluator in self.extra_evaluators.items():
            metrics[key] = evaluator(state, self.target)
        return metrics

    def _make_plots(self, state, loss_trace) -> None:
        if loss_trace is not None:
            plotting.plot_loss(
                loss_trace,
                paths.plot_path(self.name, "loss"),
                title=f"Training loss — {self.name}",
            )
        gas_states_log, lrL, lpL, n = evaluation.slice_grid_data()
        slice_fields = evaluation.compute_slice_fields(
            state, self.target, gas_states_log, n
        )
        plotting.plot_slice(
            slice_fields, lrL, lpL, paths.plot_path(self.name, "slice")
        )
        for plot_fn in self.extra_plots:
            plot_fn(self, state)
