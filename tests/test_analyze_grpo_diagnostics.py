"""Tests for the GRPO trajectory-size diagnostic report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_grpo_diagnostics import (
    load_generation_rollouts,
    summarize_rollouts,
)


class AnalyzeGrpoDiagnosticsTest(unittest.TestCase):
    def test_latest_step_is_selected_and_long_rollouts_rank_first(self):
        records = [
            {
                "event": "generation_batch",
                "global_step": 0,
                "generation_batch": 1,
                "rollouts": [{"task_id": 1, "context_max_input_tokens": 100}],
            },
            {
                "event": "generation_batch",
                "global_step": 1,
                "generation_batch": 1,
                "rollouts": [
                    {
                        "task_id": 2,
                        "context_max_input_tokens": 12000,
                        "cumulative_input_tokens": 30000,
                        "observation_raw_tokens": 5000,
                        "observation_visible_tokens": 3000,
                        "observation_truncated_count": 2,
                        "actions": [{"tool": "search"}],
                    },
                    {
                        "task_id": 3,
                        "context_max_input_tokens": 22000,
                        "cumulative_input_tokens": 70000,
                        "observation_raw_tokens": 9000,
                        "observation_visible_tokens": 4000,
                        "actions": [{"tool": "click"}],
                    },
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_diagnostics.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            step, rollouts = load_generation_rollouts(path)

        report = summarize_rollouts(step, rollouts, top=1)
        self.assertEqual(step, 1)
        self.assertEqual(report["rollout_count"], 2)
        self.assertEqual(
            report["token_fields"]["context_max_input_tokens"]["p50"], 12000
        )
        self.assertEqual(
            report["token_fields"]["context_max_input_tokens"]["p90"], 22000
        )
        self.assertEqual(report["top_by_context_length"][0]["task_id"], 3)
        self.assertEqual(
            report["top_by_context_length"][0]["observation_tokens_removed"],
            5000,
        )
        self.assertEqual(report["tool_counts"], {"click": 1, "search": 1})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
