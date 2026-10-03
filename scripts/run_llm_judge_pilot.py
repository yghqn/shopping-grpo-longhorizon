#!/usr/bin/env python3
"""Run a small resumable Rubric Curator and Trajectory Judge API pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shopping_grpo.evaluation.artifacts import (  # noqa: E402
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    load_json,
    write_json_atomic,
)
from shopping_grpo.evaluation.comparison import (  # noqa: E402
    compare_evaluation_runs,
)
from shopping_grpo.evaluation.contracts import (  # noqa: E402
    rubric_ids,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.model_client import (  # noqa: E402
    OpenAIJSONClient,
)
from shopping_grpo.evaluation.prompts import (  # noqa: E402
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (  # noqa: E402
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.rubric import (  # noqa: E402
    materialize_rubric_bundle,
)


DEFAULT_PREPARED = ROOT / "outputs/evaluation/llm-judge-dry-run-v4"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/llm-judge-pilot-10"
DEFAULT_CURATOR_MODEL = "deepseek-flash"
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
PILOT_MANIFEST_VERSION = "shopping-llm-judge-pilot-v1"


def _existing_index(path: Path, *, key: str) -> dict:
    if not path.exists():
        return {}
    return index_jsonl(path, key=key)


def _usage_totals(metadata_rows: list[Mapping]) -> dict:
    totals = {
        "requests": len(metadata_rows),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }
    for metadata in metadata_rows:
        usage = metadata.get("usage")
        if not isinstance(usage, Mapping):
            continue
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get(
            "completion_tokens", usage.get("output_tokens", 0)
        )
        total = usage.get("total_tokens", 0)
        input_details = usage.get(
            "prompt_tokens_details", usage.get("input_tokens_details", {})
        )
        output_details = usage.get("output_tokens_details", {})
        if not isinstance(input_details, Mapping):
            input_details = {}
        if not isinstance(output_details, Mapping):
            output_details = {}
        totals["prompt_tokens"] += int(prompt or 0)
        totals["completion_tokens"] += int(completion or 0)
        totals["total_tokens"] += int(total or 0)
        totals["cached_tokens"] += int(
            usage.get(
                "prompt_cache_hit_tokens",
                input_details.get("cached_tokens", 0),
            )
            or 0
        )
        totals["reasoning_tokens"] += int(
            output_details.get("reasoning_tokens", 0) or 0
        )
    return totals


def _client(
    *,
    model: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> OpenAIJSONClient:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        raise SystemExit("OPENAI_BASE_URL is required")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    return OpenAIJSONClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        response_format_json=True,
        thinking=False,
    )


def _ensure_manifest(
    *,
    output_dir: Path,
    task_ids: list[int],
    run_labels: list[str],
    curator_model: str,
    judge_model: str,
) -> None:
    path = output_dir / "pilot_manifest.json"
    expected = {
        "schema_version": PILOT_MANIFEST_VERSION,
        "task_ids": task_ids,
        "run_labels": run_labels,
        "curator": {
            "model": curator_model,
            "prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
            "thinking": False,
        },
        "judge": {
            "model": judge_model,
            "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
            "thinking": False,
        },
    }
    if path.exists():
        if load_json(path) != expected:
            raise SystemExit(
                f"pilot manifest mismatch: {path}; use a new output directory"
            )
        return
    write_json_atomic(path, expected)


def _curate(
    *,
    task_ids: list[int],
    prepared_dir: Path,
    output_dir: Path,
    model: str,
    client: OpenAIJSONClient,
) -> tuple[dict[int, dict], list[dict]]:
    shared_input = prepared_dir / "shared"
    shared_output = output_dir / "shared"
    facts = index_jsonl(shared_input / "task_facts.jsonl", key="task_id")
    candidates = index_jsonl(
        shared_input / "rubric_candidates.jsonl", key="task_id"
    )
    requests = index_jsonl(
        shared_input / "rubric_requests.jsonl", key="task_id"
    )
    response_path = shared_output / "curator_responses.jsonl"
    rubric_path = shared_output / "rubrics.jsonl"
    responses = _existing_index(response_path, key="task_id")
    rubrics = _existing_index(rubric_path, key="task_id")

    for position, task_id in enumerate(task_ids, start=1):
        if task_id in rubrics:
            validate_rubric_bundle(rubrics[task_id], expected_task_id=task_id)
            print(f"[curator {position}/{len(task_ids)}] cached task={task_id}")
            continue
        if task_id in responses:
            result = responses[task_id]["result"]
            metadata = responses[task_id].get("metadata") or {}
        else:
            print(f"[curator {position}/{len(task_ids)}] calling task={task_id}")
            completion = client.complete_json(requests[task_id]["messages"])
            result = completion["result"]
            metadata = completion["metadata"]
            response_row = {
                "task_id": task_id,
                "result": result,
                "metadata": metadata,
            }
            append_jsonl_fsync(response_path, response_row)
            responses[task_id] = response_row

        bundle = materialize_rubric_bundle(
            task_facts=facts[task_id],
            candidates=candidates[task_id],
            curator_response=result,
            curator_model=model,
            curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
            rubric_version="final-200-rubric-v1",
        )
        append_jsonl_fsync(rubric_path, bundle)
        rubrics[task_id] = bundle

    metadata_rows = [
        responses[task_id].get("metadata") or {}
        for task_id in task_ids
        if task_id in responses
    ]
    return rubrics, metadata_rows


def _judge_run(
    *,
    label: str,
    task_ids: list[int],
    prepared_dir: Path,
    output_dir: Path,
    rubrics: Mapping[int, Mapping],
    client: OpenAIJSONClient,
) -> tuple[list[dict], list[dict]]:
    preprocessed = index_jsonl(
        prepared_dir / label / "preprocessed.jsonl",
        key="task_id",
    )
    run_dir = output_dir / label
    request_path = run_dir / "judge_requests.jsonl"
    judge_path = run_dir / "judges.jsonl"
    evaluation_path = run_dir / "evaluations.jsonl"
    judges = _existing_index(judge_path, key="task_id")
    evaluations = _existing_index(evaluation_path, key="task_id")

    for position, task_id in enumerate(task_ids, start=1):
        row = preprocessed[task_id]
        normalized = row["normalized_trajectory"]
        metrics = row["deterministic_metrics"]
        rubric = rubrics[task_id]
        trajectory_id = str(normalized["trajectory_id"])

        if task_id in evaluations:
            print(
                f"[judge {label} {position}/{len(task_ids)}] "
                f"cached task={task_id}"
            )
            continue

        if task_id in judges:
            judge_result = judges[task_id]["result"]
        elif metrics["validity"]["infrastructure_invalid"]:
            judge_result = build_not_judged_result(
                task_id=task_id,
                trajectory_id=trajectory_id,
                reason="infrastructure_invalid",
            )
            append_jsonl_fsync(
                judge_path,
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "result": judge_result,
                    "metadata": {"api_called": False},
                },
            )
            judges[task_id] = {
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "result": judge_result,
                "metadata": {"api_called": False},
            }
        else:
            messages = build_trajectory_judge_messages(
                normalized=normalized,
                rubric_bundle=rubric,
                deterministic_metrics=metrics,
            )
            print(
                f"[judge {label} {position}/{len(task_ids)}] "
                f"calling task={task_id}"
            )
            completion = client.complete_json(messages)
            judge_result = validate_judge_result(
                completion["result"],
                rubric_ids=rubric_ids(rubric),
                rubric_definitions=rubric["rubrics"],
                expected_task_id=task_id,
                expected_trajectory_id=trajectory_id,
                allowed_event_ids=[
                    event["event_id"]
                    for event in normalized.get("events") or []
                    if isinstance(event, Mapping) and event.get("event_id")
                ],
            )
            append_jsonl_fsync(
                request_path,
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "messages": messages,
                },
            )
            judge_row = {
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "result": judge_result,
                "metadata": completion["metadata"],
            }
            append_jsonl_fsync(judge_path, judge_row)
            judges[task_id] = judge_row

        evaluation = assemble_task_evaluation(
            actor={"label": label},
            normalized_trajectory=normalized,
            deterministic_metrics=metrics,
            rubric_bundle=rubric,
            judge_result=judge_result,
        )
        append_jsonl_fsync(evaluation_path, evaluation)
        evaluations[task_id] = evaluation

    ordered = [evaluations[task_id] for task_id in task_ids]
    summary = summarize_evaluations(
        expected_task_ids=task_ids,
        evaluations=ordered,
    )
    write_json_atomic(
        run_dir / "evaluation_summary.json",
        summary,
        force=(run_dir / "evaluation_summary.json").exists(),
    )
    metadata_rows = [
        judges[task_id].get("metadata") or {}
        for task_id in task_ids
        if judges.get(task_id, {}).get("metadata", {}).get("api_called", True)
    ]
    return ordered, metadata_rows


def run(args: argparse.Namespace) -> dict:
    prepared_summary = load_json(args.prepared_dir / "dry_run_summary.json")
    available_labels = [row["label"] for row in prepared_summary["runs"]]
    labels = args.run or available_labels
    unknown = sorted(set(labels) - set(available_labels))
    if unknown:
        raise SystemExit(f"unknown prepared run labels: {unknown}")
    all_task_ids = [int(value) for value in prepared_summary["task_ids"]]
    if args.task_id:
        if len(set(args.task_id)) != len(args.task_id):
            raise SystemExit("--task-id values must be unique")
        missing = sorted(set(args.task_id) - set(all_task_ids))
        if missing:
            raise SystemExit(f"unknown task IDs: {missing}")
        task_ids = list(args.task_id)
    else:
        task_ids = all_task_ids[: args.max_tasks]
    _ensure_manifest(
        output_dir=args.output_dir,
        task_ids=task_ids,
        run_labels=labels,
        curator_model=args.curator_model,
        judge_model=args.judge_model,
    )

    curator_client = _client(
        model=args.curator_model,
        max_tokens=args.curator_max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    judge_client = _client(
        model=args.judge_model,
        max_tokens=args.judge_max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    rubrics, curator_metadata = _curate(
        task_ids=task_ids,
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        model=args.curator_model,
        client=curator_client,
    )

    runs = {}
    judge_metadata = []
    for label in labels:
        evaluations, metadata = _judge_run(
            label=label,
            task_ids=task_ids,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            rubrics=rubrics,
            client=judge_client,
        )
        runs[label] = evaluations
        judge_metadata.extend(metadata)

    comparison = compare_evaluation_runs(
        expected_task_ids=task_ids,
        runs=runs,
    )
    write_json_atomic(
        args.output_dir / "model_comparison.json",
        comparison,
        force=(args.output_dir / "model_comparison.json").exists(),
    )
    usage = {
        "schema_version": "shopping-llm-judge-usage-v1",
        "curator": _usage_totals(curator_metadata),
        "judge": _usage_totals(judge_metadata),
    }
    write_json_atomic(
        args.output_dir / "api_usage.json",
        usage,
        force=(args.output_dir / "api_usage.json").exists(),
    )
    return usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 10-task API pilot with frozen Rubrics and three actors."
    )
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        help="Run an explicit task ID; repeat for multiple tasks.",
    )
    parser.add_argument("--run", action="append")
    parser.add_argument("--curator-model", default=DEFAULT_CURATOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--curator-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    if args.curator_max_tokens < 1 or args.judge_max_tokens < 1:
        parser.error("max token limits must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def main() -> None:
    usage = run(parse_args())
    print(json.dumps(usage, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
