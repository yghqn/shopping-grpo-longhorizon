"""验证 benchmark 清单与评测入口的最小 CLI 行为。"""

import sys
import unittest
from unittest.mock import patch

from scripts.evaluate_shop_benchmark import parse_args


class BenchmarkCliTest(unittest.TestCase):
    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.context_window, 24576)
        self.assertEqual(args.context_safety_margin, 512)
        self.assertFalse(args.context_compaction)
        self.assertEqual(args.observation_token_budget, 1536)
        self.assertEqual(args.observation_detail_token_budget, 4096)
        self.assertEqual(args.observation_generic_token_budget, 768)
        self.assertEqual(args.observation_search_top_k, 20)
        self.assertFalse(args.candidate_memory)
        self.assertEqual(args.candidate_memory_token_budget, 1024)
        self.assertEqual(args.cumulative_input_token_budget, 0)

    def test_candidate_memory_is_explicitly_enabled(self):
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/budget_memory/dev_smoke.jsonl",
                "--output",
                "outputs/eval/memory/raw.jsonl",
                "--summary",
                "outputs/eval/memory/summary.json",
                "--model",
                "shopping-agent",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
                "--candidate-memory",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.candidate_memory)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
