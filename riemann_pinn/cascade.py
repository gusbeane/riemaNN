"""Residual cascade: compose a frozen base model with a trainable correction MLP.

The k-th stage predicts p*(x) = p_base(x) * 10**eps_k(x), where p_base is
the composition of stages 0..k-1 (frozen) and eps_k is a small MLP whose
weights are the trainable parameters of stage k.

All inference is pure feed-forward — no solver / Newton iterations.

Usage:
  # Stage 1: plain model (e.g. StarPressureMLPNormalized) trained normally.
  base_forward = lambda x: stage1_state.apply_fn({"params": stage1_state.params}, x)

  # Stage 2: train a CorrectionMLP on top of base_forward.
  cascade = CascadeModel(base_forward=base_forward, width=256, depth=3)
  # train cascade.apply normally; its trainable params are only the eps MLP.

  # After training stage 2, rebuild base_forward for stage 3:
  eps2 = CorrectionMLP(width=256, depth=3)
  base_forward = lambda x: stage1(...) * 10**eps2.apply({"params": stage2_state.params["CorrectionMLP_0"]}, x)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state as flax_train_state
from tqdm import tqdm

from . import physics
from .physics import GAS_STATE_DIM


class CorrectionMLP(nn.Module):
    """MLP that outputs a scalar log10-space correction eps(x).

    Final Dense is zero-initialized so the untrained stage produces eps=0,
    i.e. it starts as identity on the base prediction.
    """

    width: int = 256
    depth: int = 3
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = nn.Dense(self.width)(h)
            h = self.activation(h)
        eps = nn.Dense(1, kernel_init=nn.initializers.zeros)(h).squeeze(-1)
        return eps


class CascadeModel(nn.Module):
    """Training wrapper for a single cascade stage.

    forward(x) = base_forward(x) * 10**eps(x).
    Only the eps MLP has trainable parameters; base_forward is frozen.
    """

    base_forward: Callable = None
    width: int = 256
    depth: int = 3
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, x):
        p_base = jax.lax.stop_gradient(self.base_forward(x))
        eps = CorrectionMLP(
            width=self.width, depth=self.depth, activation=self.activation,
        )(x)
        return p_base * (10.0 ** eps)


# --- losses ------------------------------------------------------------------


def cascade_supervised_log_loss(params, apply_fn, gas_states_log):
    """Log-MSE against exact solver p*. Scale-invariant."""
    p_pred = apply_fn({"params": params}, gas_states_log)
    gas_phys = physics.gas_log_to_phys(gas_states_log)
    log_p_true = jnp.log10(jax.vmap(physics.find_pstar_precise)(gas_phys)[0])
    log_p_pred = jnp.log10(p_pred)
    diff = log_p_pred - log_p_true
    loss = jnp.mean(diff ** 2)
    return loss, {"loss/log_mse": loss}


def cascade_supervised_abs_loss(params, apply_fn, gas_states_log):
    """Plain (p-p_true)^2 MSE — directly targets absolute error but weights large p* more."""
    p_pred = apply_fn({"params": params}, gas_states_log)
    gas_phys = physics.gas_log_to_phys(gas_states_log)
    p_true, _ = jax.vmap(physics.find_pstar_precise)(gas_phys)
    diff = p_pred - p_true
    return jnp.mean(diff ** 2), {"loss/abs_mse": jnp.mean(diff ** 2)}


# --- train state + steps -----------------------------------------------------


def create_cascade_train_state(rng, model, optimizer, batch_size_hint=2048):
    dummy = jnp.ones((batch_size_hint, GAS_STATE_DIM))
    params = model.init(rng, dummy)["params"]
    return flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )


def make_cascade_train_step(loss_fn: Callable):
    @jax.jit
    def step(state, gas_log):
        def _w(p):
            return loss_fn(p, state.apply_fn, gas_log)
        (loss, metrics), grads = jax.value_and_grad(_w, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    return step


def make_cascade_lbfgs_step(loss_fn: Callable):
    import optax

    @jax.jit
    def step(state, gas_log):
        def value_fn(p):
            l, _ = loss_fn(p, state.apply_fn, gas_log)
            return l

        (loss, metrics), grads = jax.value_and_grad(
            lambda p: loss_fn(p, state.apply_fn, gas_log), has_aux=True,
        )(state.params)

        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params,
            value=loss, grad=grads, value_fn=value_fn,
        )
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(
            step=state.step + 1, params=new_params, opt_state=new_opt_state,
        ), loss, metrics

    return step


# --- run loop ----------------------------------------------------------------


def run_cascade_loop(
    state, step_fn, sampler, rng, *, n_epochs, batch_size,
    desc="cascade", log_every=200, split_key_every=1,
):
    import jax.random as jr

    losses = []
    pbar = tqdm(range(n_epochs), desc=desc)
    for epoch in pbar:
        if (epoch % split_key_every) == 0:
            rng, bk = jr.split(rng)
        batch = sampler(bk, batch_size)
        state, loss, _ = step_fn(state, batch)
        losses.append(float(loss))
        if epoch % log_every == 0:
            pbar.set_postfix(loss=f"{loss:.3e}")
    return state, jnp.array(losses)


# --- checkpoint I/O ----------------------------------------------------------


def save_stage(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(state))


def load_stage(path: Path, template_state):
    return from_bytes(template_state, path.read_bytes())


# --- composition for inference ----------------------------------------------


def compose_chain(stage0_forward, *eps_specs):
    """Compose a cascade chain into a single forward fn.

    stage0_forward: x -> p_base (the Stage-1 plain pstar predictor).
    eps_specs: sequence of (CorrectionMLP instance, params) tuples, in cascade order.
    Returns x -> p* for the whole chain.
    """
    def forward(x):
        p = stage0_forward(x)
        for mlp, p_eps in eps_specs:
            eps = mlp.apply({"params": p_eps}, x)
            p = p * (10.0 ** eps)
        return p
    return forward
