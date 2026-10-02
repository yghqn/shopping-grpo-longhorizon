#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/shopping-grpo-export.XXXXXX")"
RAW_EXPORT_DIR="$TEMP_DIR/raw-export"

cleanup() {
  rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "export output must not already exist: $OUTPUT_DIR" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$RAW_EXPORT_DIR" \
  --trust-remote-code

if [[ ! -f "$RAW_EXPORT_DIR/lora_adapter/adapter_model.safetensors" ]]; then
  echo "veRL export did not contain a LoRA adapter: $RAW_EXPORT_DIR" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/merge_lora_adapter.py" \
  --base-model "$RAW_EXPORT_DIR" \
  --adapter "$RAW_EXPORT_DIR/lora_adapter" \
  --output "$OUTPUT_DIR" \
  --bf16

cp -a "$RAW_EXPORT_DIR/lora_adapter" "$OUTPUT_DIR/lora_adapter"

echo "GRPO adapter merged into standalone model: $OUTPUT_DIR"
