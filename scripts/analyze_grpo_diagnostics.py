#!/usr/bin/env python3
"""Summarize the trajectory-size drivers recorded by a GRPO run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_FIELDS = (
    "context_max_input_tokens",
    "cumulative_input_tokens",
    "observation_raw_tokens",
    "observation_visible_tokens",
    "observation_max_raw_tokens",
    "observation_max_visible_tokens",
)


def percentile(values: list[int], percentile_value: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile_value) - 1))
    return ordered[index]


def load_generation_rollouts(
    path: Path, global_step: int | None = None
) -> tuple[int, list[dict[str, Any]]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if record.get("event") == "generation_batch":
            records.append(record)
    if not records:
        raise ValueError("no generation_batch records found")

    selected_step = (
        max(int(record["global_step"]) for record in records)
        if global_step is None
        else int(global_step)
    )
    rollouts = []
    for record in records:
        if int(record["global_step"]) != selected_step:
            continue
        for rollout in record.get("rollouts") or []:
            item = dict(rollout)
            item["generation_batch"] = int(record.get("generation_batch", 0))
            rollouts.append(item)
    if not rollouts:
        raise ValueError(f"no generation rollouts found for global_step={selected_step}")
    return selected_step, rollouts


def summarize_rollouts(
    global_step: int, rollouts: list[dict[str, Any]], top: int = 10
) -> dict[str, Any]:
    field_summaries = {}
    for field in TOKEN_FIELDS:
        values = [int(rollout.get(field, 0)) for rollout in rollouts]
        field_summaries[field] = {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "max": max(values, default=0),
        }

    tool_counts: Counter[str] = Counter()
    for rollout in rollouts:
        tool_counts.update(
            str(action.get("tool", "unknown"))
            for action in rollout.get("actions") or []
        )

    top_rollouts = []
    ordering = sorted(
        rollouts,
        key=lambda item: (
            int(item.get("context_max_input_tokens", 0)),
            int(item.get("cumulative_input_tokens", 0)),
        ),
        reverse=True,
    )
    for rollout in ordering[:top]:
        raw = int(rollout.get("observation_raw_tokens", 0))
        visible = int(rollout.get("observation_visible_tokens", 0))
        top_rollouts.append(
            {
                "task_id": rollout.get("task_id"),
                "uid": rollout.get("uid"),
                "rollout_index": rollout.get("rollout_index"),
                "generation_batch": rollout.get("generation_batch"),
                "steps": int(rollout.get("steps", 0)),
                "action_attempts": int(rollout.get("action_attempts", 0)),
                "termination_reason": rollout.get("termination_reason"),
                "reward_type": rollout.get("reward_type"),
                "context_max_input_tokens": int(
                    rollout.get("context_max_input_tokens", 0)
                ),
                "cumulative_input_tokens": int(
                    rollout.get("cumulative_input_tokens", 0)
                ),
                "observation_raw_tokens": raw,
                "observation_visible_tokens": visible,
                "observation_tokens_removed": max(raw - visible, 0),
                "observation_truncated_count": int(
                    rollout.get("observation_truncated_count", 0)
                ),
                "observation_max_raw_tokens": int(
                    rollout.get("observation_max_raw_tokens", 0)
                ),
                "observation_max_visible_tokens": int(
                    rollout.get("observation_max_visible_tokens", 0)
                ),
            }
        )

    return {
        "global_step": global_step,
        "rollout_count": len(rollouts),
        "generation_batches": sorted(
            {int(rollout.get("generation_batch", 0)) for rollout in rollouts}
        ),
        "token_fields": field_summaries,
        "tool_counts": dict(sorted(tool_counts.items())),
        "top_by_context_length": top_rollouts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--global-step", type=int)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step, rollouts = load_generation_rollouts(args.diagnostics, args.global_step)
    print(json.dumps(summarize_rollouts(step, rollouts, args.top), indent=2))


if __name__ == "__main__":
    main()
