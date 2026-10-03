import unittest

from scripts.run_llm_judge_pilot import _usage_totals


class LlmJudgePilotTests(unittest.TestCase):
    def test_usage_totals_support_chat_completion_shape(self):
        totals = _usage_totals(
            [
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "prompt_cache_hit_tokens": 30,
                    }
                }
            ]
        )
        self.assertEqual(totals["requests"], 1)
        self.assertEqual(totals["prompt_tokens"], 100)
        self.assertEqual(totals["completion_tokens"], 20)
        self.assertEqual(totals["cached_tokens"], 30)

    def test_usage_totals_support_responses_shape(self):
        totals = _usage_totals(
            [
                {
                    "usage": {
                        "input_tokens": 80,
                        "output_tokens": 15,
                        "total_tokens": 95,
                        "input_tokens_details": {"cached_tokens": 10},
                        "output_tokens_details": {"reasoning_tokens": 5},
                    }
                }
            ]
        )
        self.assertEqual(totals["prompt_tokens"], 80)
        self.assertEqual(totals["completion_tokens"], 15)
        self.assertEqual(totals["cached_tokens"], 10)
        self.assertEqual(totals["reasoning_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
