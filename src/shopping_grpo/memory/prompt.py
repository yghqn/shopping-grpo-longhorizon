"""Prompt renderers shared by evaluation and online GRPO rollouts."""

from __future__ import annotations

from shopping_grpo.environment.actions import clickable_buttons, product_ids


TRAINING_MEMORY_CONTEXT_START = "[SHOPPING_TRAINING_MEMORY_CONTEXT_V1]"
TRAINING_MEMORY_CONTEXT_END = "[END_SHOPPING_TRAINING_MEMORY_CONTEXT_V1]"


def render_current_action_state(observation: object) -> str:
    """Render the only targets that are legal for the actor's next tool call."""
    observation = str(observation or "")
    asins = product_ids(observation)
    buttons = clickable_buttons(observation)
    lines = [
        "[CURRENT_ACTION_STATE_V1]",
        "这是下一步工具参数的唯一合法来源；历史候选只能用于比较。",
        "当前可打开商品 ASIN: " + (", ".join(asins[:20]) if asins else "无"),
        "当前可点击按钮: " + (", ".join(buttons[:30]) if buttons else "无"),
    ]
    normalized = {button.casefold() for button in buttons}
    if "< prev" in normalized:
        lines.append(
            "当前位于信息子页；若继续核验当前商品，下一步优先调用 prev_page 返回商品详情。"
        )
    return "\n".join(lines)


def compose_memory_system_content(
    base_content: object,
    memory_content: object,
    observation: object,
) -> str:
    """Merge dynamic memory into the single leading system message."""
    sections = [
        str(base_content or "").strip(),
        render_current_action_state(observation),
        str(memory_content or "").strip(),
    ]
    return "\n\n".join(section for section in sections if section)


def compose_training_tool_content(
    observation: object,
    memory_content: object,
) -> str:
    """Attach one replaceable memory capsule to the newest tool observation."""
    capsule = "\n\n".join(
        (
            TRAINING_MEMORY_CONTEXT_START,
            render_current_action_state(observation),
            str(memory_content or "").strip(),
            TRAINING_MEMORY_CONTEXT_END,
        )
    )
    return "\n\n".join((str(observation or "").strip(), capsule))
