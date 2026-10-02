"""Regression contract for serving an exported GRPO actor."""

from pathlib import Path


def test_export_grpo_merges_the_separate_verl_lora_adapter():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "export_grpo.sh").read_text(encoding="utf-8")

    assert "verl.model_merger" in script
    assert 'RAW_EXPORT_DIR="$TEMP_DIR/raw-export"' in script
    assert '"$RAW_EXPORT_DIR/lora_adapter"' in script
    assert '"$ROOT/scripts/merge_lora_adapter.py"' in script
    assert 'cp -a "$RAW_EXPORT_DIR/lora_adapter"' in script
    assert "GRPO adapter merged into standalone model" in script
