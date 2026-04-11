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


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "star_pressure_mlp": StarPressureMLP,
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
