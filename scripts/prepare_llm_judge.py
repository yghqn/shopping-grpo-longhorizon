#!/usr/bin/env python3
"""Prepare frozen Rubric and trajectory-Judge inputs without network or GPU.

This command performs only deterministic, local work.  It reconstructs private
TaskFacts from the frozen ShopSimulator corpus, extracts code-owned Rubric
candidates, normalizes saved actor trajectories, and writes request payloads
for a later API-backed Curator/Judge run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gzip
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
for import_root in (SRC, SHOP_ENV):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from shopping_grpo.evaluation.artifacts import (  # noqa: E402
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.metrics import (  # noqa: E402
    compute_deterministic_metrics,
)
from shopping_grpo.evaluation.prompts import (  # noqa: E402
    RUBRIC_CURATOR_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (  # noqa: E402
    extract_rubric_candidates,
)
from shopping_grpo.evaluation.task_facts import (  # noqa: E402
    task_facts_from_product_corpus,
)
from shopping_grpo.evaluation.trajectory import (  # noqa: E402
    normalize_trajectory,
)
from web_agent_site.engine.constraints import (  # noqa: E402
    explicit_budget_from_instruction,
)
from web_agent_site.engine.reward_features import (  # noqa: E402
    compile_reward_features,
)


DEFAULT_TASKS = ROOT / "data/evaluation/tasks.jsonl"
DEFAULT_PRODUCTS = (
    SHOP_ENV / "data/fine_items_eval_train_all.json.gz"
)
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/llm-judge-dry-run-v4"
_INSTRUCTION_PREFIX = re.compile(r"^\s*Instruction:\s*", re.I)


def _load_products(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    if not isinstance(products, list):
        raise ValueError("product corpus root must be a list")
    return products


def _load_task_ids(path: Path, *, limit: int | None) -> list[int]:
    task_ids = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task manifest contains duplicate task IDs")
    if limit is not None:
        task_ids = task_ids[:limit]
    if not task_ids:
        raise ValueError("task manifest selected no tasks")
    return task_ids


def _reward_goal(instruction: Mapping, product: Mapping) -> dict:
    goal = dict(compile_reward_features(instruction, product))
    goal["category"] = product.get("category")
    goal["price_upper"] = explicit_budget_from_instruction(
        str(instruction.get("instruction") or "")
    )
    return goal


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    label = label.strip()
    if not separator or not label or not path.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=TRAJECTORIES.jsonl")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise argparse.ArgumentTypeError(
            "run label may contain only letters, digits, dot, underscore, and dash"
        )
    return label, Path(path)


def _canonical_query(value: object) -> str:
    return _INSTRUCTION_PREFIX.sub("", str(value or "").strip())


def _preprocess_run(
    *,
    label: str,
    trajectories_path: Path,
    task_ids: list[int],
    facts_by_task: Mapping[int, Mapping],
    output_dir: Path,
    force: bool,
) -> dict:
    expected = set(task_ids)
    by_task = {}
    for raw in iter_jsonl(trajectories_path):
        task_id = int(raw["task_id"])
        if task_id not in expected:
            raise ValueError(
                f"{label} contains unexpected task_id {task_id}"
            )
        if task_id in by_task:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        normalized = normalize_trajectory(raw)
        expected_query = _canonical_query(facts_by_task[task_id]["query"])
        actual_query = _canonical_query(normalized.get("actor_query"))
        if actual_query != expected_query:
            raise ValueError(f"{label} task {task_id} Query does not match TaskFacts")
        metrics = compute_deterministic_metrics(normalized)
        by_task[task_id] = {
            "task_id": task_id,
            "trajectory_id": normalized["trajectory_id"],
            "normalized_trajectory": normalized,
            "deterministic_metrics": metrics,
        }

    missing = sorted(expected - set(by_task))
    if missing:
        raise ValueError(f"{label} is missing task IDs: {missing}")
    rows = [by_task[task_id] for task_id in task_ids]
    run_dir = output_dir / label
    write_jsonl_atomic(run_dir / "preprocessed.jsonl", rows, force=force)
    eligible = sum(
        not row["deterministic_metrics"]["validity"]["infrastructure_invalid"]
        for row in rows
    )
    return {
        "label": label,
        "source": str(trajectories_path),
        "trajectories": len(rows),
        "judge_eligible": eligible,
        "not_judge_eligible": len(rows) - eligible,
        "preprocessed": str(run_dir / "preprocessed.jsonl"),
    }


def prepare(args: argparse.Namespace) -> dict:
    task_ids = _load_task_ids(args.tasks, limit=args.limit)
    products = _load_products(args.products)
    facts = task_facts_from_product_corpus(
        task_ids=task_ids,
        products=products,
        reward_goal_builder=_reward_goal,
    )
    facts_by_task = {int(row["task_id"]): row for row in facts}
    candidates = [extract_rubric_candidates(row) for row in facts]
    curator_requests = [
        {
            "task_id": int(row["task_id"]),
            "prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
            "messages": build_rubric_curator_messages(
                task_id=int(row["task_id"]),
                query=str(row["query"]),
                candidates=list(candidate["candidates"]),
            ),
        }
        for row, candidate in zip(facts, candidates, strict=True)
    ]

    output_dir = args.output_dir
    shared_dir = output_dir / "shared"
    write_jsonl_atomic(shared_dir / "task_facts.jsonl", facts, force=args.force)
    write_jsonl_atomic(
        shared_dir / "rubric_candidates.jsonl",
        candidates,
        force=args.force,
    )
    write_jsonl_atomic(
        shared_dir / "rubric_requests.jsonl",
        curator_requests,
        force=args.force,
    )

    run_summaries = []
    labels = set()
    for label, path in args.run:
        if label in labels:
            raise ValueError(f"duplicate run label: {label}")
        labels.add(label)
        run_summaries.append(
            _preprocess_run(
                label=label,
                trajectories_path=path,
                task_ids=task_ids,
                facts_by_task=facts_by_task,
                output_dir=output_dir,
                force=args.force,
            )
        )

    candidate_counts = [len(row["candidates"]) for row in candidates]
    summary = {
        "schema_version": "shopping-llm-judge-dry-run-v1",
        "network_calls": 0,
        "gpu_required": False,
        "tasks": len(task_ids),
        "task_ids": task_ids,
        "rubric_requests": len(curator_requests),
        "rubric_candidate_count": sum(candidate_counts),
        "mean_rubric_candidates_per_task": (
            sum(candidate_counts) / len(candidate_counts)
        ),
        "runs": run_summaries,
        "next_required_artifact": "shared/rubrics.jsonl",
    }
    write_json_atomic(output_dir / "dry_run_summary.json", summary, force=args.force)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TaskFacts, Rubric Curator requests, normalized trajectories, "
            "and deterministic metrics without calling an LLM API."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    summary = prepare(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
