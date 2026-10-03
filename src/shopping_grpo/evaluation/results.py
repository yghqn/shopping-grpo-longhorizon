"""Assemble four evaluation sections and paired-ready summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy

from shopping_grpo.evaluation.contracts import (
    CONTRACT_VERSION,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    rubric_ids,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.metrics import DETERMINISTIC_METRICS_VERSION
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION


EVALUATION_RESULT_VERSION = "shopping-per-task-evaluation-v1"
EVALUATION_SUMMARY_VERSION = "shopping-evaluation-summary-v1"


def build_not_judged_result(
    *,
    task_id: int,
    trajectory_id: str,
    reason: str,
) -> dict:
    """Represent an infrastructure-invalid trajectory without fake zero scores."""

    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": int(task_id),
        "trajectory_id": str(trajectory_id),
        "judge_status": "not_judged",
        "rubric_assessments": [],
        "dimension_scores": {},
        "errors": {
            "primary": str(reason),
            "secondary": [],
            "evidence_event_ids": [],
        },
        "overall_diagnosis": (
            "轨迹因基础设施无效而未进行需求和轨迹质量评分。"
        ),
    }


def _reward_rubric_disagreement(
    *,
    metrics: Mapping,
    rubric_bundle: Mapping,
    judge_result: Mapping,
) -> dict:
    if judge_result.get("judge_status") != "valid":
        return {"value": False, "reasons": []}
    hardness = {
        item["rubric_id"]: item["hardness"]
        for item in rubric_bundle["rubrics"]
    }
    statuses = {
        item["rubric_id"]: item["status"]
        for item in judge_result["rubric_assessments"]
    }
    reward = metrics.get("reward_and_outcome")
    reward = reward if isinstance(reward, Mapping) else {}
    reward_type = reward.get("reward_type")
    purchase_success = reward.get("purchase_success") is True
    reasons = []

    violated = sorted(
        rubric_id
        for rubric_id, status in statuses.items()
        if status == "violated"
    )
    violated_hard = sorted(
        rubric_id
        for rubric_id in violated
        if hardness.get(rubric_id) == "hard"
    )
    if purchase_success and violated:
        reasons.append(
            {
                "type": "reward_success_with_rubric_violation",
                "rubric_ids": violated,
            }
        )
    if purchase_success and violated_hard:
        reasons.append(
            {
                "type": "reward_success_with_hard_rubric_violation",
                "rubric_ids": violated_hard,
            }
        )

    applicable_hard = {
        rubric_id: status
        for rubric_id, status in statuses.items()
        if hardness.get(rubric_id) == "hard"
        and status != "not_applicable"
    }
    all_hard_satisfied = bool(applicable_hard) and all(
        status == "satisfied" for status in applicable_hard.values()
    )
    if reward_type == "wrong_purchase" and all_hard_satisfied:
        reasons.append(
            {
                "type": "wrong_purchase_with_all_hard_rubrics_satisfied",
                "rubric_ids": sorted(applicable_hard),
            }
        )
    applicable = {
        rubric_id: status
        for rubric_id, status in statuses.items()
        if status != "not_applicable"
    }
    if (
        reward_type == "partial_alternative_purchase"
        and applicable
        and all(status == "satisfied" for status in applicable.values())
    ):
        reasons.append(
            {
                "type": "partial_reward_with_all_rubrics_satisfied",
                "rubric_ids": sorted(applicable),
            }
        )
    return {"value": bool(reasons), "reasons": reasons}


def assemble_task_evaluation(
    *,
    actor: Mapping,
    normalized_trajectory: Mapping,
    deterministic_metrics: Mapping,
    rubric_bundle: Mapping,
    judge_result: Mapping,
) -> dict:
    """Join one Actor run without mixing its four evaluation sections."""

    if normalized_trajectory.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError("unsupported normalized trajectory schema")
    if deterministic_metrics.get("schema_version") != DETERMINISTIC_METRICS_VERSION:
        raise ValueError("unsupported deterministic metrics schema")
    task_id = int(normalized_trajectory["task_id"])
    trajectory_id = str(normalized_trajectory["trajectory_id"])
    rubric = validate_rubric_bundle(
        rubric_bundle,
        expected_task_id=task_id,
    )
    judge = validate_judge_result(
        judge_result,
        rubric_ids=rubric_ids(rubric),
        rubric_definitions=rubric["rubrics"],
        expected_task_id=task_id,
        expected_trajectory_id=trajectory_id,
        allowed_event_ids=[
            event["event_id"]
            for event in normalized_trajectory.get("events") or []
            if isinstance(event, Mapping) and event.get("event_id")
        ],
    )
    if int(deterministic_metrics.get("task_id")) != task_id:
        raise ValueError("metrics task_id does not match trajectory")
    if str(deterministic_metrics.get("trajectory_id")) != trajectory_id:
        raise ValueError("metrics trajectory_id does not match trajectory")

    validity = deterministic_metrics.get("validity")
    validity = validity if isinstance(validity, Mapping) else {}
    if validity.get("infrastructure_invalid") and judge["judge_status"] != "not_judged":
        raise ValueError(
            "infrastructure-invalid trajectories must use judge_status=not_judged"
        )
    disagreement = _reward_rubric_disagreement(
        metrics=deterministic_metrics,
        rubric_bundle=rubric,
        judge_result=judge,
    )
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "actor": deepcopy(dict(actor)),
        "reward_and_terminal": {
            "metrics": deepcopy(
                deterministic_metrics["reward_and_outcome"]
            ),
            "terminal": deepcopy(normalized_trajectory.get("terminal") or {}),
        },
        "requirement_rubric": {
            "rubric_version": rubric["rubric_version"],
            "rubrics": deepcopy(rubric["rubrics"]),
            "assessments": deepcopy(judge["rubric_assessments"]),
            "reward_rubric_disagreement": disagreement["value"],
            "disagreement_reasons": disagreement["reasons"],
        },
        "trajectory_quality": {
            "judge_status": judge["judge_status"],
            "dimension_scores": deepcopy(judge["dimension_scores"]),
            "errors": deepcopy(judge["errors"]),
            "overall_diagnosis": judge["overall_diagnosis"],
        },
        "deterministic": {
            key: deepcopy(value)
            for key, value in deterministic_metrics.items()
            if key
            not in {
                "schema_version",
                "evaluation_contract",
                "trajectory_id",
                "task_id",
                "reward_and_outcome",
            }
        },
        "artifacts": {
            "normalized_trajectory_schema": NORMALIZED_TRAJECTORY_VERSION,
            "deterministic_metrics_schema": DETERMINISTIC_METRICS_VERSION,
            "rubric_schema": rubric["schema_version"],
            "judge_schema": judge["schema_version"],
        },
    }


def _mean(total: float, denominator: int) -> float:
    return total / denominator if denominator else 0.0


def summarize_evaluations(
    *,
    expected_task_ids: Iterable[int],
    evaluations: Iterable[Mapping],
) -> dict:
    """Summarize four panels without producing a composite score."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    expected_set = set(expected)
    by_task = {}
    unexpected = []
    for record in evaluations:
        if record.get("schema_version") != EVALUATION_RESULT_VERSION:
            raise ValueError("unsupported per-task evaluation schema")
        task_id = int(record["task_id"])
        if task_id not in expected_set:
            unexpected.append(task_id)
            continue
        if task_id in by_task:
            raise ValueError(f"duplicate evaluation for task_id {task_id}")
        by_task[task_id] = record
    if unexpected:
        raise ValueError(f"unexpected task_ids: {sorted(set(unexpected))}")

    denominator = len(expected)
    missing = sorted(expected_set - set(by_task))
    reward_type_counts = Counter()
    strict_successes = []
    purchase_successes = 0
    reward_valid = 0
    total_reward = 0.0
    total_terminal_utility = 0.0
    total_weighted_score = 0.0
    rubric_status = Counter()
    rubric_status_by_hardness = defaultdict(Counter)
    disagreement_tasks = []
    judge_status = Counter()
    dimension_distributions = {
        name: Counter() for name in JUDGE_DIMENSIONS
    }
    primary_errors = Counter()
    secondary_errors = Counter()
    primary_error_task_ids = defaultdict(list)
    secondary_error_task_ids = defaultdict(list)
    total_steps = 0
    total_attempts = 0
    total_guards = 0
    total_duplicate_actions = 0
    total_duplicate_searches = 0
    truncated_tasks = 0
    infrastructure_invalid_tasks = []

    for task_id, record in by_task.items():
        reward = record["reward_and_terminal"]["metrics"]
        reward_type_counts[str(reward.get("reward_type") or "unknown")] += 1
        if reward.get("strict_gold_success") is True:
            strict_successes.append(task_id)
        purchase_successes += reward.get("purchase_success") is True
        reward_valid += reward.get("reward_valid") is True
        total_reward += float(reward.get("final_reward", 0.0) or 0.0)
        total_terminal_utility += float(
            reward.get("terminal_utility", 0.0) or 0.0
        )
        total_weighted_score += float(
            reward.get("weighted_score", 0.0) or 0.0
        )

        rubric = record["requirement_rubric"]
        hardness = {
            item["rubric_id"]: item["hardness"]
            for item in rubric["rubrics"]
        }
        for assessment in rubric["assessments"]:
            status = assessment["status"]
            rubric_status[status] += 1
            rubric_status_by_hardness[
                hardness.get(assessment["rubric_id"], "unknown")
            ][status] += 1
        if rubric["reward_rubric_disagreement"]:
            disagreement_tasks.append(task_id)

        quality = record["trajectory_quality"]
        judge_status[quality["judge_status"]] += 1
        if quality["judge_status"] == "valid":
            for name in JUDGE_DIMENSIONS:
                dimension_distributions[name][
                    int(quality["dimension_scores"][name]["score"])
                ] += 1
            primary = quality["errors"].get("primary")
            if primary:
                primary_errors[str(primary)] += 1
                primary_error_task_ids[str(primary)].append(task_id)
            for secondary in quality["errors"].get("secondary") or []:
                secondary_errors[str(secondary)] += 1
                secondary_error_task_ids[str(secondary)].append(task_id)

        deterministic = record["deterministic"]
        actions = deterministic["actions_and_efficiency"]
        repetition = deterministic["repetition"]
        legality = deterministic["legality"]
        context = deterministic["context"]
        validity = deterministic["validity"]
        total_steps += int(actions.get("executed_tool_steps", 0))
        total_attempts += int(actions.get("action_attempts", 0))
        total_guards += int(legality.get("guard_rejection_count", 0))
        total_duplicate_actions += int(
            repetition.get("duplicate_canonical_action_count", 0)
        )
        total_duplicate_searches += int(
            repetition.get("duplicate_search_query_count", 0)
        )
        truncated_tasks += bool(context.get("any_observation_truncated"))
        if validity.get("infrastructure_invalid"):
            infrastructure_invalid_tasks.append(task_id)

    valid_judges = judge_status.get("valid", 0)
    dimension_summary = {}
    for name, distribution in dimension_distributions.items():
        score_total = sum(score * count for score, count in distribution.items())
        dimension_summary[name] = {
            "score_counts": {
                str(score): distribution.get(score, 0) for score in (0, 1, 2)
            },
            "mean_score_among_valid_judges": _mean(
                score_total, valid_judges
            ),
        }
    return {
        "schema_version": EVALUATION_SUMMARY_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": denominator,
        "completed_evaluations": len(by_task),
        "missing_task_ids": missing,
        "reward_and_terminal": {
            "strict_gold_successes": len(strict_successes),
            "strict_gold_task_ids": sorted(strict_successes),
            "gold_purchase_rate": _mean(len(strict_successes), denominator),
            "purchase_successes": purchase_successes,
            "purchase_success_rate": _mean(
                purchase_successes, denominator
            ),
            "reward_valid_tasks": reward_valid,
            "reward_valid_rate": _mean(reward_valid, denominator),
            "reward_type_counts": dict(sorted(reward_type_counts.items())),
            "total_final_reward": total_reward,
            "mean_final_reward_fixed_denominator": _mean(
                total_reward, denominator
            ),
            "mean_terminal_utility_fixed_denominator": _mean(
                total_terminal_utility, denominator
            ),
            "mean_weighted_score_fixed_denominator": _mean(
                total_weighted_score, denominator
            ),
        },
        "requirement_rubric": {
            "status_counts": dict(sorted(rubric_status.items())),
            "status_counts_by_hardness": {
                hardness: dict(sorted(counts.items()))
                for hardness, counts in sorted(
                    rubric_status_by_hardness.items()
                )
            },
            "reward_rubric_disagreement_tasks": len(disagreement_tasks),
            "reward_rubric_disagreement_task_ids": sorted(
                disagreement_tasks
            ),
        },
        "trajectory_quality": {
            "judge_status_counts": dict(sorted(judge_status.items())),
            "judge_coverage_rate": _mean(valid_judges, denominator),
            "dimensions": dimension_summary,
            "primary_error_counts": dict(sorted(primary_errors.items())),
            "primary_error_task_ids": {
                error: sorted(task_ids)
                for error, task_ids in sorted(primary_error_task_ids.items())
            },
            "secondary_error_counts": dict(
                sorted(secondary_errors.items())
            ),
            "secondary_error_task_ids": {
                error: sorted(task_ids)
                for error, task_ids in sorted(secondary_error_task_ids.items())
            },
        },
        "deterministic": {
            "average_executed_steps_fixed_denominator": _mean(
                total_steps, denominator
            ),
            "average_action_attempts_fixed_denominator": _mean(
                total_attempts, denominator
            ),
            "total_guard_rejections": total_guards,
            "total_duplicate_canonical_actions": total_duplicate_actions,
            "total_duplicate_search_queries": total_duplicate_searches,
            "tasks_with_observation_truncation": truncated_tasks,
            "infrastructure_invalid_tasks": len(
                infrastructure_invalid_tasks
            ),
            "infrastructure_invalid_task_ids": sorted(
                infrastructure_invalid_tasks
            ),
        },
    }
