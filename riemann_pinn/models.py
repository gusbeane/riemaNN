"""Flax neural network modules and a build-from-spec factory.

Experiment scripts should import model classes directly, e.g.
`from riemann_pinn.models import StarPressureMLP` and construct them
with keyword arguments. The `MODEL_REGISTRY` + `build_model(spec)`
path exists only for the comparison script, which needs to rehydrate
a model from the config.json snapshot on disk.
"""

from __future__ import annotations

from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp

from . import physics
from .physics import GAS_STATE_DIM


class StarPressureMLP(nn.Module):
    """MLP that maps a log-space gas state to a scalar (default: log10(p*))."""

    width: int = 64
    depth: int = 2
    activation: Callable = nn.silu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_dim)(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x


class TwoRarefactionMLP(nn.Module):
    """MLP with two-rarefaction p0 appended as 6th input.

    Accepts (B, 5) log-space gas state, computes log10(p0) internally,
    and passes (B, 6) to an inner MLP. External interface is unchanged.
    """

    width: int = 256
    depth: int = 3
    activation: Callable = nn.silu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        gas_phys = physics.gas_log_to_phys(x)
        p0 = physics.two_rarefaction_p0_batch(gas_phys)
        log_p0 = jnp.log10(jnp.maximum(p0, 1e-30))
        x_aug = jnp.concatenate([x, log_p0[:, None]], axis=-1)
        for _ in range(self.depth):
            x_aug = nn.Dense(self.width)(x_aug)
            x_aug = self.activation(x_aug)
        x_aug = nn.Dense(self.output_dim)(x_aug)
        if self.output_dim == 1:
            x_aug = x_aug.squeeze(-1)
        return x_aug


class SymmetricMLP(nn.Module):
    """MLP that enforces Riemann L/R symmetry by construction.

    Output = 0.5 * (g(qL, qR, uRL) + g(qR, qL, -uRL))
    where g is a shared StarPressureMLP sub-module.
    """

    width: int = 256
    depth: int = 3
    activation: Callable = nn.silu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        inner = StarPressureMLP(
            width=self.width,
            depth=self.depth,
            activation=self.activation,
            output_dim=self.output_dim,
            name="inner",
        )
        x_swap = jnp.concatenate(
            [x[:, 2:4], x[:, 0:2], -x[:, 4:5]], axis=-1
        )
        return 0.5 * (inner(x) + inner(x_swap))


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "star_pressure_mlp": StarPressureMLP,
    "two_rarefaction_mlp": TwoRarefactionMLP,
    "symmetric_mlp": SymmetricMLP,
}


def model_spec(model: nn.Module) -> dict[str, Any]:
    """Return a JSON-serializable spec that `build_model` can rehydrate."""
    name = None
    for key, cls in MODEL_REGISTRY.items():
        if type(model) is cls:
            name = key
            break
    if name is None:
        raise ValueError(
            f"Model {type(model).__name__} is not in MODEL_REGISTRY; add it "
            f"if you want this experiment to be loadable from config.json."
        )
    spec: dict[str, Any] = {"type": name}
    # Only include JSON-safe fields; skip activation (a callable) unless it is nn.silu etc.
    for field in type(model).__dataclass_fields__:  # type: ignore[attr-defined]
        if field in ("parent", "name"):
            continue
        value = getattr(model, field)
        if callable(value):
            # Record the activation by name; build_model will look it up.
            spec[field] = getattr(value, "__name__", None)
        else:
            spec[field] = value
    return spec


_ACTIVATION_BY_NAME: dict[str, Callable] = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "tanh": nn.tanh,
}


def build_model(spec: dict[str, Any] | nn.Module) -> nn.Module:
    """Construct a Flax module from a spec dict, or pass through an existing module."""
    if isinstance(spec, nn.Module):
        return spec
    spec = dict(spec)
    type_name = spec.pop("type")
    cls = MODEL_REGISTRY[type_name]
    if "activation" in spec and isinstance(spec["activation"], str):
        spec["activation"] = _ACTIVATION_BY_NAME[spec["activation"]]
    return cls(**spec)


def dummy_input(batch_size: int = 1):
    """Shape hint for Flax `init` on any gas-state model."""
    import jax.numpy as jnp

    return jnp.ones((batch_size, GAS_STATE_DIM))
