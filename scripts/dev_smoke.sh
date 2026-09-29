#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-baseline}"

case "$ARM" in
  baseline)
    MEMORY_ARGS=()
    ;;
  memory)
    MEMORY_ARGS=(--candidate-memory --candidate-memory-token-budget 1024)
    ;;
  *)
    echo "Usage: bash scripts/dev_smoke.sh [baseline|memory]" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$ROOT/outputs/budget-memory/dev-smoke/$ARM"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_shop_benchmark.py" \
  --benchmark "$ROOT/data/budget_memory/dev_smoke.jsonl" \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url http://127.0.0.1:5700 \
  --model shopping-agent \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0.0 \
  --top-p 1.0 \
  --context-window 24576 \
  --context-safety-margin 512 \
  --max-tokens 512 \
  --cumulative-input-token-budget 0 \
  "${MEMORY_ARGS[@]}"
