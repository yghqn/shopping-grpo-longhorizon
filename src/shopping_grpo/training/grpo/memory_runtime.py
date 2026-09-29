"""Candidate-memory lifecycle helpers shared by online GRPO workers."""

from __future__ import annotations

from collections.abc import Mapping

from shopping_grpo.memory import CandidateMemory, compose_training_tool_content


def initialize_candidate_memory(
    *,
    requirement: object,
    initial_observation_state: Mapping | None,
    token_budget: int,
    count_tokens,
) -> tuple[CandidateMemory, dict]:
    """Create one public-observation memory and its initial audit view."""
    memory = CandidateMemory(requirement=str(requirement or ""))
    if initial_observation_state is not None:
        memory.observe(initial_observation_state, step_index=0, tool_name="reset")
    view = memory.render_view(token_budget=int(token_budget), count_tokens=count_tokens)
    return memory, memory_view_diagnostics(view, step_index=-1)


def update_candidate_memory(
    memory: CandidateMemory,
    *,
    observation_state: Mapping,
    observation: object,
    step_index: int,
    tool_name: str,
    token_budget: int,
    count_tokens,
) -> tuple[str, dict]:
    """Update memory from public state and render one replaceable tool capsule."""
    memory.observe(
        observation_state,
        step_index=int(step_index),
        tool_name=str(tool_name),
    )
    view = memory.render_view(token_budget=int(token_budget), count_tokens=count_tokens)
    content = compose_training_tool_content(observation, view.content)
    return content, memory_view_diagnostics(view, step_index=int(step_index))


def memory_view_diagnostics(view, *, step_index: int) -> dict:
    """Keep audit hashes and selection counts without duplicating prompt text."""
    payload = view.to_dict()
    payload.pop("content", None)
    payload["step_index"] = int(step_index)
    payload["selected_count"] = len(payload.pop("selected_asins"))
    payload["dropped_count"] = len(payload.pop("dropped_asins"))
    return payload
