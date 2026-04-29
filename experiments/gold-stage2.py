"""Smoke test: short AdamW run to verify the 3D delta-space pipeline.

Minimal working example and the default target for
`venv/bin/python run.py experiments/smoke_test.py`.
"""

import optax
import jax.numpy as jnp

from riemann_pinn.model import PressureMLP
from riemann_pinn.train import Experiment, Phase, residual_loss, DataSet
from run import load_experiments, load_checkpoint, build_template_state
from pathlib import Path


_DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)

N_EPOCHS = 500
BATCH_SIZE = 2**16 # about 65k
LR = 2e-3

# generate residuals
import numpy as np
print('loading data...', end='', flush=True)
data = np.load("data/train_10M.npz")
print('done')
gas_states = data["gas_states"]
pstar = data["pstar"]

gold_exp = load_experiments(Path("experiments/gold.py"))[0]
ckpt_path = Path("outputs/gold/gold/checkpoint.msgpack")
state = load_checkpoint(ckpt_path, build_template_state(gold_exp))

print("applying model...", end='', flush=True)
pstar_nn = state.apply_fn({"params": state.params}, gas_states)
print('done')
residuals = pstar/pstar_nn
eps = jnp.max(residuals)
residuals /= eps

dataset = DataSet(gas_states=gas_states, targets=residuals)

experiments = [
    Experiment(
        name=f"gold",
        model=PressureMLP(width=16, depth=2),
        domain=_DOMAIN,
        seed=42,
        prev_stages=[(state, eps)],
        phases=[
            Phase(
                tx=optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(
                        learning_rate=LR,
                        # optax.cosine_decay_schedule(lr, N_EPOCHS, alpha=1e-7),
                    ),
                ),
                n_epochs=N_EPOCHS,
                loss=residual_loss,
                batch_size=BATCH_SIZE,
                sampler=dataset,
                log_every=1,
                name="adamw",
            ),
        ],
    )
]

