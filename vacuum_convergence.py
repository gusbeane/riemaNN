"""Resolution convergence study for Test 3 (vacuum generation)
with the exact Godunov solver."""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from euler1D import (solve, make_ic_riemann, exact_ref, TESTS,
                     FLOOR, GAMMA, GM1)

cfg = TESTS["3_vacuum"]
resolutions = [50, 100, 200, 400, 800]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
labels = [r"$\rho$", r"$u$", r"$p$"]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(resolutions)))

# analytic reference on fine grid
x_fine = np.linspace(cfg["domain"][0], cfg["domain"][1], 2000)
dx_fine = x_fine[1] - x_fine[0]
_, rf, uf, pf = exact_ref(x_fine, cfg["x0"], cfg["t_end"],
                           cfg["wL"], cfg["wR"])
for ax, lab, ref in zip(axes, labels, [rf, uf, pf]):
    ax.plot(x_fine, ref, "k-", lw=2, alpha=0.35, label="Analytic")

for N, col in zip(resolutions, colors):
    x, dx, rho0, u0, p0 = make_ic_riemann(
        N, cfg["domain"], cfg["x0"], cfg["wL"], cfg["wR"])
    print(f"N={N:4d} ... ", end="", flush=True)
    r, u, p = solve(rho0, u0, p0, dx, cfg["t_end"], solver="exact")
    print("done")
    for ax, arr in zip(axes, [r, u, p]):
        ax.plot(x, arr, color=col, lw=1.0, label=f"N={N}")

for ax, lab in zip(axes, labels):
    ax.set_xlabel("x")
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc="best")

fig.suptitle("Test 3 – vacuum: exact Godunov convergence", fontsize=12, y=1.02)
fig.tight_layout()
out = "./outputs/3_vacuum_convergence.png"
os.makedirs("./outputs", exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
