#!/usr/bin/env bash
# Run every experiment in a list-valued experiment file, capped at N in parallel.
#
#   ./run_grid.sh experiments/adamw_normmlp_wdgrid.py            # 4 in parallel
#   ./run_grid.sh experiments/adamw_normmlp_wdgrid.py 2          # 2 in parallel
#
# Shows a fixed layout: one line with total progress + one line per running
# worker (latest tqdm frame). Stdout/stderr of each worker is captured to a
# temp log file (removed on exit).

set -euo pipefail

EXPERIMENT=${1:?usage: $0 <experiment.py> [max_parallel]}
MAX_PARALLEL=${2:-4}

if [[ ! -f "$EXPERIMENT" ]]; then
  echo "error: $EXPERIMENT not found" >&2
  exit 1
fi

N=$(venv/bin/python run.py "$EXPERIMENT" --count)
if [[ "$N" -le 1 ]]; then
  echo "only $N experiment in $EXPERIMENT; run directly with:"
  echo "  venv/bin/python run.py $EXPERIMENT"
  exit 1
fi
if [[ "$MAX_PARALLEL" -gt "$N" ]]; then
  MAX_PARALLEL=$N
fi

LOG_DIR=$(mktemp -d -t rungrid.XXXXXX)
STAMP_DIR="$LOG_DIR/done"
mkdir -p "$STAMP_DIR"

cleanup() {
  # Restore cursor, kill renderer if still alive, remove logs
  tput cnorm 2>/dev/null || true
  if [[ -n "${RENDERER_PID:-}" ]]; then
    kill "$RENDERER_PID" 2>/dev/null || true
  fi
  rm -rf "$LOG_DIR"
}
trap cleanup EXIT INT TERM

echo "Running $N experiments from $EXPERIMENT (max $MAX_PARALLEL parallel)"
echo "Logs: $LOG_DIR"
echo

# Reserve MAX_PARALLEL+1 lines for the live display.
RESERVE=$((MAX_PARALLEL + 1))
for ((i = 0; i < RESERVE; i++)); do echo; done

render() {
  tput civis 2>/dev/null || true
  local cols
  cols=$(tput cols 2>/dev/null || echo 120)
  while :; do
    local done_n running_n=0 active_logs=()
    done_n=$(find "$STAMP_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    for f in "$LOG_DIR"/run_*.log; do
      [[ -f "$f" ]] || continue
      local idx=${f##*/run_}
      idx=${idx%.log}
      if [[ ! -f "$STAMP_DIR/$idx" ]]; then
        active_logs+=("$f")
        running_n=$((running_n + 1))
      fi
    done

    tput cuu "$RESERVE"
    printf '\r\033[K[%d/%d done, %d running]\n' "$done_n" "$N" "$running_n"

    local slot=0
    for f in "${active_logs[@]}"; do
      local idx=${f##*/run_}
      idx=${idx%.log}
      local last
      last=$(tr '\r' '\n' < "$f" 2>/dev/null | awk 'NF' | tail -n 1)
      local line
      line=$(printf '  [%s] %s' "$idx" "$last")
      printf '\r\033[K%.*s\n' "$cols" "$line"
      slot=$((slot + 1))
    done
    while [[ $slot -lt $MAX_PARALLEL ]]; do
      printf '\r\033[K  .\n'
      slot=$((slot + 1))
    done

    if [[ "$done_n" -ge "$N" ]]; then
      return 0
    fi
    sleep 0.5
  done
}

render &
RENDERER_PID=$!

# Fan out indices 0..N-1 to xargs with -P MAX_PARALLEL.
# Each worker writes its stdout+stderr to its own log and touches a stamp on exit.
seq 0 $((N - 1)) | xargs -I{} -P "$MAX_PARALLEL" bash -c '
  idx=$0
  log_dir=$1
  stamp_dir=$2
  exp=$3
  log="$log_dir/run_$idx.log"
  venv/bin/python run.py "$exp" --index "$idx" >"$log" 2>&1 || true
  touch "$stamp_dir/$idx"
' {} "$LOG_DIR" "$STAMP_DIR" "$EXPERIMENT"

wait "$RENDERER_PID" 2>/dev/null || true

# Summarize failures by scanning logs for non-JSON tails.
failed=()
for ((i = 0; i < N; i++)); do
  log="$LOG_DIR/run_$i.log"
  if ! grep -q '"median_abs_fstar"' "$log" 2>/dev/null; then
    failed+=("$i")
  fi
done
if ((${#failed[@]})); then
  echo
  echo "WARNING: ${#failed[@]} run(s) did not produce metrics: ${failed[*]}"
  echo "Inspect logs under $LOG_DIR (kept for debugging)"
  trap - EXIT
  tput cnorm 2>/dev/null || true
  exit 1
fi

echo
echo "All $N runs complete."
echo
venv/bin/python run.py "$EXPERIMENT" --print-metrics
