#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q \
  tests/test_verl_adapter.py \
  tests/test_grpo_dynamic_sampling_config.py \
  tests/test_context_window.py \
  tests/test_candidate_memory.py \
  tests/test_grpo_memory_runtime.py \
  tests/test_grpo_ablation.py \
  tests/test_experiment_config.py \
  tests/test_verl_dynamic_sampling.py

"$PYTHON_BIN" scripts/run_experiment.py \
  grpo_budget_memory_v1 \
  --dry-run

echo "Budget-memory GRPO integration validation passed."
