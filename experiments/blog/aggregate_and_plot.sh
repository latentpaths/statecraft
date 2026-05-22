#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIGS=(
  experiments/blog/configs/easy.json
  experiments/blog/configs/medium.json
  experiments/blog/configs/hard.json
  experiments/blog/configs/v_hard.json
)

PER_MODEL=0
AGGREGATE_ONLY=0
PLOT_ONLY=0

usage() {
  cat <<'EOF'
Usage: experiments/blog/aggregate_and_plot.sh [options]

Run aggregate.py and plot.py for each blog grid config.

Options:
  --per-model       also write per-model figures (plot.py --per-model)
  --aggregate-only  skip plot.py
  --plot-only       skip aggregate.py
  -h, --help        show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --per-model) PER_MODEL=1 ;;
    --aggregate-only) AGGREGATE_ONLY=1 ;;
    --plot-only) PLOT_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ "$AGGREGATE_ONLY" -eq 1 && "$PLOT_ONLY" -eq 1 ]]; then
  echo "Choose at most one of --aggregate-only and --plot-only." >&2
  exit 1
fi

run_aggregate() {
  for config in "${CONFIGS[@]}"; do
    echo "=== aggregate: $config ==="
    uv run python experiments/blog/aggregate.py --config "$config"
  done
}

run_plot() {
  for config in "${CONFIGS[@]}"; do
    echo "=== plot: $config ==="
    if [[ "$PER_MODEL" -eq 1 ]]; then
      uv run experiments/blog/plot.py --per-model --config "$config"
    else
      uv run experiments/blog/plot.py --config "$config"
    fi
  done
}

if [[ "$PLOT_ONLY" -eq 0 ]]; then
  run_aggregate
fi

if [[ "$AGGREGATE_ONLY" -eq 0 ]]; then
  run_plot
fi
