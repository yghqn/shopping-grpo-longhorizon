#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-all}"
BENCHMARK="$ROOT/data/budget_memory/dev_holdout_40.jsonl"
OUTPUT_ROOT="$ROOT/outputs/budget-memory/dev-holdout-40"

run_arm() {
  local arm="$1"
  local cumulative_budget=0
  local -a memory_args=()

  case "$arm" in
    baseline)
      ;;
    budget-only)
      cumulative_budget=100000
      ;;
    memory-v4)
      cumulative_budget=100000
      memory_args=(--candidate-memory --candidate-memory-token-budget 1024)
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      return 2
      ;;
  esac

  local output_dir="$OUTPUT_ROOT/$arm"
  mkdir -p "$output_dir"
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_shop_benchmark.py" \
    --benchmark "$BENCHMARK" \
    --output "$output_dir/trajectories.jsonl" \
    --summary "$output_dir/summary.json" \
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
    --cumulative-input-token-budget "$cumulative_budget" \
    "${memory_args[@]}" \
    2>&1 | tee "$output_dir/run.log"
}

compare_runs() {
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/compare_budget_memory_runs.py" \
    --benchmark "$BENCHMARK" \
    --run "baseline=$OUTPUT_ROOT/baseline" \
    --run "budget-only=$OUTPUT_ROOT/budget-only" \
    --run "memory-v4=$OUTPUT_ROOT/memory-v4" \
    --reference baseline \
    --output-json "$OUTPUT_ROOT/comparison.json" \
    --output-markdown "$OUTPUT_ROOT/comparison.md"
}

case "$ARM" in
  all)
    run_arm baseline
    run_arm budget-only
    run_arm memory-v4
    compare_runs
    ;;
  compare)
    compare_runs
    ;;
  baseline|budget-only|memory-v4)
    run_arm "$ARM"
    ;;
  *)
    echo "Usage: bash scripts/dev_holdout.sh [all|baseline|budget-only|memory-v4|compare]" >&2
    exit 2
    ;;
esac
