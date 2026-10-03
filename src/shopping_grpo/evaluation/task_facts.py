"""Map ShopSimulator goals to private versioned TaskFacts records."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from shopping_grpo.evaluation.rubric import build_task_facts


def task_facts_from_environment(
    *,
    task_ids: Iterable[int],
    goals: list[Mapping],
    product_item_dict: Mapping,
) -> list[dict]:
    """Build private TaskFacts in requested task order without running an env."""

    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids contains duplicates")
    rows = []
    for task_id in requested:
        if task_id < 0 or task_id >= len(goals):
            raise IndexError(
                f"task_id {task_id} is outside goal range [0, {len(goals)})"
            )
        goal = goals[task_id]
        if not isinstance(goal, Mapping):
            raise ValueError(f"goal {task_id} must be an object")
        asin = goal.get("asin")
        product = product_item_dict.get(asin)
        if not isinstance(product, Mapping):
            raise ValueError(
                f"goal {task_id} target ASIN {asin!r} is missing"
            )
        query = str(goal.get("instruction_text") or "").strip()
        if not query:
            raise ValueError(f"goal {task_id} has no instruction_text")
        instruction_record = {
            "instruction": query,
            "attributes": goal.get("attributes") or [],
            "instruction_options": goal.get("goal_options") or [],
        }
        rows.append(
            build_task_facts(
                task_id=task_id,
                query=query,
                target_product=product,
                instruction_record=instruction_record,
                reward_goal=goal,
            )
        )
    return rows


def task_facts_from_product_corpus(
    *,
    task_ids: Iterable[int],
    products: list[Mapping],
    reward_goal_builder: Callable[[Mapping, Mapping], Mapping] | None = None,
) -> list[dict]:
    """Build TaskFacts from the frozen product corpus used by Final-200.

    Final-200 task IDs are product-corpus indices and every selected product has
    exactly one frozen instruction.  ``reward_goal_builder`` lets the caller use
    the repository's Environment v2.1 reward-feature compiler without making
    the reusable evaluation package import the embedded simulator directly.
    """

    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids contains duplicates")

    rows = []
    for task_id in requested:
        if task_id < 0 or task_id >= len(products):
            raise IndexError(
                f"task_id {task_id} is outside product range "
                f"[0, {len(products)})"
            )
        product = products[task_id]
        if not isinstance(product, Mapping):
            raise ValueError(f"product {task_id} must be an object")
        instructions = product.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != 1:
            raise ValueError(
                f"product {task_id} must contain exactly one instruction"
            )
        instruction = instructions[0]
        if not isinstance(instruction, Mapping):
            raise ValueError(f"product {task_id} instruction must be an object")
        query = str(instruction.get("instruction") or "").strip()
        if not query:
            raise ValueError(f"product {task_id} instruction is empty")

        reward_goal = (
            reward_goal_builder(instruction, product)
            if reward_goal_builder is not None
            else {}
        )
        if not isinstance(reward_goal, Mapping):
            raise ValueError(
                f"reward goal builder returned a non-object for task {task_id}"
            )
        rows.append(
            build_task_facts(
                task_id=task_id,
                query=query,
                target_product=product,
                instruction_record=instruction,
                reward_goal=reward_goal,
            )
        )
    return rows
