import unittest

from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    validate_judge_result,
)
from shopping_grpo.evaluation.prompts import TRAJECTORY_JUDGE_SYSTEM_PROMPT
from shopping_grpo.evaluation.results import (
    EVALUATION_RESULT_VERSION,
    summarize_evaluations,
)


def _judge_result(*, primary, secondary):
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": 1,
        "trajectory_id": "trajectory-1",
        "judge_status": "valid",
        "rubric_assessments": [],
        "dimension_scores": {
            name: {"score": 1, "reason": "ok", "evidence_event_ids": []}
            for name in JUDGE_DIMENSIONS
        },
        "errors": {
            "primary": primary,
            "secondary": secondary,
            "evidence_event_ids": [],
        },
        "overall_diagnosis": "diagnosis",
    }


def _evaluation(task_id, *, primary, secondary):
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "task_id": task_id,
        "reward_and_terminal": {
            "metrics": {
                "strict_gold_success": False,
                "purchase_success": False,
                "reward_valid": True,
            }
        },
        "requirement_rubric": {
            "rubrics": [],
            "assessments": [],
            "reward_rubric_disagreement": False,
        },
        "trajectory_quality": {
            "judge_status": "valid",
            "dimension_scores": {
                name: {"score": 1} for name in JUDGE_DIMENSIONS
            },
            "errors": {"primary": primary, "secondary": secondary},
        },
        "deterministic": {
            "actions_and_efficiency": {},
            "repetition": {},
            "legality": {},
            "context": {},
            "validity": {},
        },
    }


class BadcaseContractTests(unittest.TestCase):
    def test_prompt_limits_secondary_errors_to_two(self):
        self.assertIn("最多两个次要错误", TRAJECTORY_JUDGE_SYSTEM_PROMPT)

    def test_rejects_more_than_two_secondary_errors(self):
        result = _judge_result(
            primary="critical_evidence_missing",
            secondary=[
                "wrong_option",
                "premature_purchase",
                "repeat_loop",
            ],
        )

        with self.assertRaisesRegex(
            ContractValidationError,
            "at most two values",
        ):
            validate_judge_result(result, rubric_ids=[])

    def test_rejects_secondary_errors_without_primary_error(self):
        result = _judge_result(primary=None, secondary=["wrong_option"])

        with self.assertRaisesRegex(
            ContractValidationError,
            "must be empty when primary is null",
        ):
            validate_judge_result(result, rubric_ids=[])

    def test_rejects_perfect_decision_with_hard_violation(self):
        result = _judge_result(primary=None, secondary=[])
        result["rubric_assessments"] = [
            {
                "rubric_id": "r0001",
                "status": "violated",
                "reason": "wrong option",
                "evidence_event_ids": [],
            }
        ]
        result["dimension_scores"]["decision_quality"]["score"] = 2

        with self.assertRaisesRegex(
            ContractValidationError,
            "decision_quality cannot be 2",
        ):
            validate_judge_result(
                result,
                rubric_ids=["r0001"],
                rubric_definitions=[
                    {"rubric_id": "r0001", "hardness": "hard"}
                ],
            )

    def test_requires_primary_error_for_hard_violation(self):
        result = _judge_result(primary=None, secondary=[])
        result["rubric_assessments"] = [
            {
                "rubric_id": "r0001",
                "status": "violated",
                "reason": "wrong option",
                "evidence_event_ids": [],
            }
        ]

        with self.assertRaisesRegex(
            ContractValidationError,
            "errors.primary is required",
        ):
            validate_judge_result(
                result,
                rubric_ids=["r0001"],
                rubric_definitions=[
                    {"rubric_id": "r0001", "hardness": "hard"}
                ],
            )

    def test_summary_indexes_task_ids_by_error_type(self):
        summary = summarize_evaluations(
            expected_task_ids=[2, 1],
            evaluations=[
                _evaluation(
                    2,
                    primary="critical_evidence_missing",
                    secondary=["wrong_option"],
                ),
                _evaluation(
                    1,
                    primary="critical_evidence_missing",
                    secondary=["repeat_loop"],
                ),
            ],
        )

        quality = summary["trajectory_quality"]
        self.assertEqual(
            quality["primary_error_task_ids"],
            {"critical_evidence_missing": [1, 2]},
        )
        self.assertEqual(
            quality["secondary_error_task_ids"],
            {"repeat_loop": [1], "wrong_option": [2]},
        )


if __name__ == "__main__":
    unittest.main()
