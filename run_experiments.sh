#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    experiments/fiducial_ds.toml
    experiments/ds_epochs10k.toml
    experiments/ds_batch512.toml
    experiments/ds_lr4e-3.toml
    experiments/ds_phi6.toml
    experiments/ds_depth4.toml
)

pids=()
for cfg in "${CONFIGS[@]}"; do
    echo "Starting: $cfg"
    JAX_PLATFORMS=cpu venv/bin/python run.py --config "$cfg" --retrain &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} experiments..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "All experiments done."
