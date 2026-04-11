"""riemann_pinn: library for training PINNs on the 1D Euler Riemann problem.

Typical usage in an experiment script:

    from riemann_pinn import Experiment, models, samplers, losses, targets

    exp = Experiment(
        name="my_experiment",
        model=models.StarPressureMLP(width=64, depth=2),
        target=targets.STAR_PRESSURE_LOG10,
        sampler=samplers.uniform_log,
        loss_impl=losses.residual_loss,
    )
    exp.run()
"""

from . import (
    evaluation,
    losses,
    models,
    paths,
    physics,
    plotting,
    samplers,
    targets,
    training,
)
from .experiment import Experiment, ExperimentResult

__all__ = [
    "Experiment",
    "ExperimentResult",
    "evaluation",
    "losses",
    "models",
    "paths",
    "physics",
    "plotting",
    "samplers",
    "targets",
    "training",
]
