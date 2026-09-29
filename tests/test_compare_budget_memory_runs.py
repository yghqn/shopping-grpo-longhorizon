import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_budget_memory_runs import build_comparison, render_markdown


def trajectory(task_id, *, success=False, status="invalid_action_limit", tokens=0, blocked=0):
    terminal = {}
    done = False
    reward = 0.0
    if success:
        status = "done"
        done = True
        reward = 1.0
        terminal = {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        }
    return {
        "task_id": task_id,
        "status": status,
        "done": done,
        "final_reward": reward,
        "terminal_result": terminal,
        "steps": [{"reward": 0.0}],
        "blocked_tool_calls": [{"reason": "test"}] * blocked,
        "context_turn_tokens": [
            {"input_tokens": tokens, "cumulative_input_tokens": tokens}
        ],
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class BudgetMemoryComparisonTest(unittest.TestCase):
    def test_builds_fixed_denominator_paired_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            write_jsonl(benchmark, [{"task_id": task_id} for task_id in range(1, 5)])
            baseline = root / "baseline"
            memory = root / "memory-v4"
            write_jsonl(
                baseline / "trajectories.jsonl",
                [
                    trajectory(1, success=True, tokens=10),
                    trajectory(2, tokens=20, blocked=2),
                    trajectory(3, status="done", tokens=30),
                    trajectory(4, tokens=40, blocked=1),
                ],
            )
            write_jsonl(
                memory / "trajectories.jsonl",
                [
                    trajectory(1, success=True, tokens=8),
                    trajectory(2, success=True, tokens=25),
                    trajectory(
                        3,
                        status="cumulative_input_budget_exceeded",
                        tokens=15,
                    ),
                    trajectory(4, tokens=30),
                ],
            )

            result = build_comparison(
                benchmark=benchmark,
                runs=[("baseline", baseline), ("memory-v4", memory)],
                reference_name="baseline",
                bootstrap_samples=200,
                seed=17,
            )

        candidate = result["runs"][1]
        paired = result["comparisons"][0]
        self.assertEqual(result["fixed_denominator"], 4)
        self.assertEqual(candidate["summary"]["strict_successes"], 2)
        self.assertEqual(paired["strict_success_rate_delta"], 0.25)
        self.assertEqual(
            paired["strict_success_transitions"]["candidate_only"], [2]
        )
        self.assertEqual(paired["total_context_input_token_delta"], -22)
        self.assertEqual(paired["guard_rejection_delta"], -3)
        self.assertIn("memory-v4 vs baseline", render_markdown(result))

    def test_rejects_unexpected_task_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            write_jsonl(benchmark, [{"task_id": 1}])
            baseline = root / "baseline"
            memory = root / "memory"
            write_jsonl(baseline / "trajectories.jsonl", [trajectory(1)])
            write_jsonl(memory / "trajectories.jsonl", [trajectory(2)])

            with self.assertRaisesRegex(ValueError, "unexpected task IDs"):
                build_comparison(
                    benchmark=benchmark,
                    runs=[("baseline", baseline), ("memory", memory)],
                    reference_name="baseline",
                    bootstrap_samples=10,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
