"""Generate a fixed training set of (gas_state_log, pstar) pairs.

Samples uniformly in the wider training domain from
experiments_hipres/adamw_normmlp_geom_widen.py and computes exact p* with
the Newton/bisection solver in riemann_pinn.physics. Writes a single .npz
and prints the SHA256 of the file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from riemann_pinn import physics
from riemann_pinn.train import UniformSampler


DOMAIN = dict(
    drho_range=(-0.9, 0.9),
    dp_range=(-0.9, 0.9),
    du_range=(-3.0, 0.5),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(
    n_samples: int,
    seed: int,
    chunk: int,
    domain: dict,
) -> tuple[np.ndarray, np.ndarray]:
    find_pstar_batch = jax.jit(jax.vmap(physics.find_pstar))
    xs = np.empty((n_samples, physics.GAS_STATE_DIM), dtype=np.float64)
    ys = np.empty((n_samples,), dtype=np.float64)
    rng = jr.PRNGKey(seed)
    t0 = time.time()
    for start in tqdm(range(0, n_samples, chunk)):
        end = min(start + chunk, n_samples)
        rng, key = jr.split(rng)
        sampler = UniformSampler(**domain)
        batch = sampler.draw_batch(key, end - start)
        pstar, _ = find_pstar_batch(batch)
        xs[start:end] = np.asarray(batch)
        ys[start:end] = np.asarray(pstar)
        done = end
        rate = done / max(time.time() - t0, 1e-6)
        # print(f"  {done:>10d}/{n_samples}  ({rate:,.0f} samples/s)")
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-samples", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1 << 18)
    ap.add_argument("--drho-range", type=float, nargs=2,
                    default=DOMAIN["drho_range"])
    ap.add_argument("--dp-range", type=float, nargs=2,
                    default=DOMAIN["dp_range"])
    ap.add_argument("--du-range", type=float, nargs=2,
                    default=DOMAIN["du_range"])
    ap.add_argument("-o", "--out", type=Path,
                    default=Path(__file__).parent / "train_10M.npz")
    args = ap.parse_args()

    domain = dict(
        drho_range=tuple(args.drho_range),
        dp_range=tuple(args.dp_range),
        du_range=tuple(args.du_range),
    )
    print(f"Generating {args.n_samples:,} samples in {domain} (seed={args.seed})")
    xs, ys = generate(args.n_samples, args.seed, args.chunk, domain)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, gas_states=xs, pstar=ys, **{
        f"domain_{k}": np.asarray(v) for k, v in domain.items()
    })
    digest = _sha256(args.out)
    sha_path = args.out.with_name(args.out.name + "-sha256")
    sha_path.write_text(f"{digest}  {args.out.name}\n")
    print(f"wrote  {args.out}  ({args.out.stat().st_size / 1e9:.2f} GB)")
    print(f"wrote  {sha_path}  ({digest})")


if __name__ == "__main__":
    main()
