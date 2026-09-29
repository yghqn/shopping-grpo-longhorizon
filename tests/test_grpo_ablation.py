import unittest

from scripts.check_grpo_runtime import ppo_gradient_accumulation_steps
from shopping_grpo.training.grpo.adapter.runtime import (
    apply_reward_input_token_shaping,
    apply_reward_length_shaping,
)
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
)


class GrpoAblationTest(unittest.TestCase):
    def test_ppo_mini_and_micro_batches_define_gradient_accumulation(self):
        self.assertEqual(ppo_gradient_accumulation_steps(4, 2), 2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ppo_gradient_accumulation_steps(3, 2)

    def test_disabled_length_shaping_is_a_no_op(self):
        reward = {"total": 0.8, "terminal_utility": 0.8, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 25, "max_steps": 35, "termination_reason": "gold_purchase"}

        shaped = apply_reward_length_shaping(reward, state, enabled=False)

        self.assertEqual(shaped, reward)
        self.assertIsNot(shaped, reward)

    def test_penalty_starts_after_soft_threshold_and_is_capped(self):
        reward = {"total": 0.8, "terminal_utility": 0.8, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 30, "max_steps": 35, "termination_reason": "gold_purchase"}

        shaped = apply_reward_length_shaping(
            reward,
            state,
            enabled=True,
            soft_threshold=20,
            penalty_per_step=0.02,
            max_penalty=0.1,
        )

        self.assertAlmostEqual(shaped["penalty_overlong"], 0.1)
        self.assertAlmostEqual(shaped["total"], 0.7)
        self.assertAlmostEqual(shaped["terminal_utility"], 0.7)
        self.assertFalse(shaped["sampling_invalid"])

    def test_max_step_trajectory_is_invalid_for_dynamic_resampling(self):
        reward = {"total": 0.0, "terminal_utility": 0.0, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 35, "max_steps": 35, "termination_reason": "max_steps"}
        shaped = apply_reward_length_shaping(
            reward,
            state,
            enabled=True,
            soft_threshold=20,
            penalty_per_step=0.01,
            max_penalty=0.2,
        )
        info = {
            "overlong": shaped["overlong"],
            "infrastructure_invalid": False,
            "reward_unverifiable": False,
            "reward": {**shaped, "purchase_success": False},
        }

        _, _, invalid, reasons = extract_shopping_group_signals([info])

        self.assertEqual(invalid, [True])
        self.assertEqual(reasons, [("overlong",)])

    def test_input_token_penalty_starts_after_threshold_and_is_capped(self):
        reward = {
            "total": 0.8,
            "terminal_utility": 0.8,
            "penalty_input_tokens": 0.0,
        }

        shaped = apply_reward_input_token_shaping(
            reward,
            {"cumulative_input_tokens": 85000},
            enabled=True,
            soft_threshold=60000,
            penalty_per_1k=0.01,
            max_penalty=0.1,
        )

        self.assertAlmostEqual(shaped["penalty_input_tokens"], 0.1)
        self.assertAlmostEqual(shaped["total"], 0.7)
        self.assertAlmostEqual(shaped["terminal_utility"], 0.7)

    def test_metrics_keep_overlong_repeat_loop_and_max_step_rates(self):
        base_reward = {
            "full": 0.0, "strict": 0.0, "native": 0.0, "semantic": 0.0,
            "total": 0.0, "terminal_utility": 0.0, "efficiency": 0.0,
            "penalty_overlong": 0.0, "penalty_unfinished": 0.0, "penalty_repeat": 0.0,
            "penalty_input_tokens": 0.0,
            "repeat_action_rate": 1.0, "purchase_success": False, "sampling_invalid": True,
            "r_type": 0.0, "r_att": 0.0, "r_option": 0.0, "r_price": 0.0,
        }
        metrics = aggregate_shopping_metrics([
            {
                "steps": 35,
                "done": True,
                "overlong": True,
                "termination_reason": "max_steps",
                "reward_type": "repeat_loop",
                "infrastructure_invalid": False,
                "reward": base_reward,
            }
        ])

        self.assertEqual(metrics["trajectory/overlong_rate"], 1.0)
        self.assertEqual(metrics["trajectory/repeat_loop_rate"], 1.0)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 1.0)
        self.assertEqual(metrics["penalty/input_tokens_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
