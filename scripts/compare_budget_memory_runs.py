#!/usr/bin/env python3
"""Build a paired, fixed-denominator comparison for budget-memory runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories


SCHEMA_VERSION = "budget-memory-comparison-v1"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _task_ids(path: Path) -> list[int]:
    task_ids = [int(row["task_id"]) for row in _read_jsonl(path)]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"benchmark contains duplicate task IDs: {path}")
    return task_ids


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use NAME=DIRECTORY")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name or not directory.strip():
        raise argparse.ArgumentTypeError("--run must use non-empty NAME=DIRECTORY")
    return name, Path(directory)


def _indexed_trajectories(path: Path, expected: set[int]) -> dict[int, dict]:
    indexed = {}
    unexpected = []
    for row in _read_jsonl(path):
        task_id = int(row["task_id"])
        if task_id not in expected:
            unexpected.append(task_id)
            continue
        if task_id in indexed:
            raise ValueError(f"duplicate trajectory for task {task_id}: {path}")
        indexed[task_id] = row
    if unexpected:
        raise ValueError(f"unexpected task IDs in {path}: {sorted(set(unexpected))}")
    return indexed


def _cumulative_input_tokens(trajectory: dict) -> int:
    turns = trajectory.get("context_turn_tokens") or []
    return max((int(turn.get("cumulative_input_tokens", 0)) for turn in turns), default=0)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        return [0.0, 0.0]
    rate = successes / total
    scale = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / scale
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / scale
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
    )


def _bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[generator.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _run_record(name: str, directory: Path, task_ids: list[int]) -> dict:
    trajectory_path = directory / "trajectories.jsonl"
    indexed = _indexed_trajectories(trajectory_path, set(task_ids))
    trajectories = [indexed[task_id] for task_id in task_ids if task_id in indexed]
    summary = summarize_trajectories(task_ids, trajectories)
    success_ids = set(summary["strict_success_task_ids"])
    task_records = {}
    for task_id in task_ids:
        trajectory = indexed.get(task_id)
        task_records[str(task_id)] = {
            "present": trajectory is not None,
            "strict_success": task_id in success_ids,
            "status": trajectory.get("status") if trajectory else "missing",
            "done": bool(trajectory and trajectory.get("done")),
            "final_reward": float(trajectory.get("final_reward", 0.0))
            if trajectory
            else 0.0,
            "steps": len(trajectory.get("steps") or []) if trajectory else 0,
            "guard_rejections": len(trajectory.get("blocked_tool_calls") or [])
            if trajectory
            else 0,
            "cumulative_input_tokens": _cumulative_input_tokens(trajectory)
            if trajectory
            else 0,
        }
    summary_path = directory / "summary.json"
    saved_protocol = {}
    if summary_path.exists():
        saved_protocol = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "protocol", {}
        )
    return {
        "name": name,
        "directory": str(directory),
        "protocol": saved_protocol,
        "summary": summary,
        "strict_success_rate_ci95": _wilson_interval(
            summary["strict_successes"], len(task_ids)
        ),
        "tasks": task_records,
    }


def _paired_comparison(
    reference: dict,
    candidate: dict,
    task_ids: list[int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    reference_tasks = reference["tasks"]
    candidate_tasks = candidate["tasks"]
    candidate_only = []
    reference_only = []
    both_success = []
    neither_success = []
    success_deltas = []
    token_deltas = []
    paired_ids = []
    for task_id in task_ids:
        key = str(task_id)
        before = reference_tasks[key]
        after = candidate_tasks[key]
        before_success = bool(before["strict_success"])
        after_success = bool(after["strict_success"])
        success_deltas.append(float(after_success) - float(before_success))
        if before_success and after_success:
            both_success.append(task_id)
        elif before_success:
            reference_only.append(task_id)
        elif after_success:
            candidate_only.append(task_id)
        else:
            neither_success.append(task_id)
        if before["present"] and after["present"]:
            paired_ids.append(task_id)
            token_deltas.append(
                float(after["cumulative_input_tokens"] - before["cumulative_input_tokens"])
            )

    reference_summary = reference["summary"]
    candidate_summary = candidate["summary"]
    reference_tokens = reference_summary["context_projection"]["total_context_input_tokens"]
    candidate_tokens = candidate_summary["context_projection"]["total_context_input_tokens"]
    return {
        "reference": reference["name"],
        "candidate": candidate["name"],
        "paired_token_tasks": len(paired_ids),
        "strict_success_transitions": {
            "candidate_only": candidate_only,
            "reference_only": reference_only,
            "both_success": both_success,
            "neither_success": neither_success,
        },
        "strict_success_rate_delta": sum(success_deltas) / len(task_ids),
        "strict_success_rate_delta_ci95": _bootstrap_mean_ci(
            success_deltas,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "total_context_input_token_delta": candidate_tokens - reference_tokens,
        "total_context_input_token_delta_rate": (
            (candidate_tokens - reference_tokens) / reference_tokens
            if reference_tokens
            else 0.0
        ),
        "mean_paired_input_token_delta": (
            sum(token_deltas) / len(token_deltas) if token_deltas else 0.0
        ),
        "mean_paired_input_token_delta_ci95": _bootstrap_mean_ci(
            token_deltas,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "guard_rejection_delta": (
            candidate_summary["context_projection"]["guard_rejections"]
            - reference_summary["context_projection"]["guard_rejections"]
        ),
        "done_task_delta": candidate_summary["done_tasks"] - reference_summary["done_tasks"],
        "budget_exhaustion_delta": (
            candidate_summary["context_projection"][
                "cumulative_input_budget_exhausted_tasks"
            ]
            - reference_summary["context_projection"][
                "cumulative_input_budget_exhausted_tasks"
            ]
        ),
    }


def build_comparison(
    *,
    benchmark: Path,
    runs: list[tuple[str, Path]],
    reference_name: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    task_ids = _task_ids(benchmark)
    names = [name for name, _ in runs]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")
    if reference_name not in names:
        raise ValueError(f"reference run is missing: {reference_name}")
    records = [_run_record(name, directory, task_ids) for name, directory in runs]
    by_name = {record["name"]: record for record in records}
    reference = by_name[reference_name]
    comparisons = []
    for index, record in enumerate(records):
        if record["name"] == reference_name:
            continue
        comparisons.append(
            _paired_comparison(
                reference,
                record,
                task_ids,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index * 10,
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "path": str(benchmark),
            "sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            "task_count": len(task_ids),
            "task_ids": task_ids,
        },
        "fixed_denominator": len(task_ids),
        "missing_tasks_count_as_failures": True,
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "runs": records,
        "comparisons": comparisons,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _rate_interval(values: list[float]) -> str:
    return f"[{100 * values[0]:.1f}%, {100 * values[1]:.1f}%]"


def _percentage_points(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def _point_interval(values: list[float]) -> str:
    return f"[{100 * values[0]:.1f}, {100 * values[1]:.1f}] pp"


def _numeric_interval(values: list[float]) -> str:
    return f"[{values[0]:.1f}, {values[1]:.1f}]"


def render_markdown(comparison: dict) -> str:
    lines = [
        "# Budget-memory paired holdout comparison",
        "",
        f"Benchmark: `{comparison['benchmark']['path']}`  ",
        f"SHA256: `{comparison['benchmark']['sha256']}`  ",
        (
            f"Fixed denominator: {comparison['fixed_denominator']} tasks. "
            "Missing tasks count as failures."
        ),
        "",
        "## Aggregate results",
        "",
        "| Run | Strict success (95% CI) | Done | Invalid action | Budget exhausted | "
        "Guard rejects | Total input tokens | Mean tokens/task | Avg steps | Mean reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in comparison["runs"]:
        summary = run["summary"]
        projection = summary["context_projection"]
        strict = (
            f"{summary['strict_successes']}/{summary['expected_tasks']} "
            f"({_percent(summary['strict_success_rate'])}; "
            f"{_rate_interval(run['strict_success_rate_ci95'])})"
        )
        lines.append(
            "| {name} | {strict} | {done} | {invalid} | {budget} | {guard} | "
            "{tokens:,} | {mean_tokens:,.1f} | {steps:.2f} | {reward:.4f} |".format(
                name=run["name"],
                strict=strict,
                done=summary["done_tasks"],
                invalid=summary["status_counts"].get("invalid_action_limit", 0),
                budget=projection["cumulative_input_budget_exhausted_tasks"],
                guard=projection["guard_rejections"],
                tokens=projection["total_context_input_tokens"],
                mean_tokens=projection["mean_context_input_tokens_per_completed_task"],
                steps=summary["average_steps"],
                reward=summary["mean_final_reward"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Strict delta (bootstrap 95% CI) | Candidate-only wins | "
            "Reference-only wins | Total-token delta | Mean paired-token delta (95% CI) | "
            "Guard delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in comparison["comparisons"]:
        transitions = item["strict_success_transitions"]
        lines.append(
            "| {candidate} vs {reference} | {delta} ({ci}) | {candidate_only} | "
            "{reference_only} | {token_delta:+,} ({token_rate:+.1%}) | {mean_delta:+,.1f} "
            "({token_ci}) | {guard_delta:+d} |".format(
                candidate=item["candidate"],
                reference=item["reference"],
                delta=_percentage_points(item["strict_success_rate_delta"]),
                ci=_point_interval(item["strict_success_rate_delta_ci95"]),
                candidate_only=len(transitions["candidate_only"]),
                reference_only=len(transitions["reference_only"]),
                token_delta=item["total_context_input_token_delta"],
                token_rate=item["total_context_input_token_delta_rate"],
                mean_delta=item["mean_paired_input_token_delta"],
                token_ci=_numeric_interval(item["mean_paired_input_token_delta_ci95"]),
                guard_delta=item["guard_rejection_delta"],
            )
        )
    lines.extend(["", "## Strict-success task IDs", ""])
    for run in comparison["runs"]:
        task_ids = run["summary"]["strict_success_task_ids"]
        rendered = ", ".join(str(task_id) for task_id in task_ids) or "none"
        lines.append(f"- **{run['name']}**: {rendered}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_run_spec, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(
        benchmark=args.benchmark,
        runs=args.run,
        reference_name=args.reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
