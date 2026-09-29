"""Deterministic candidate memory built only from public ShopSimulator state.

The memory records what the agent actually observed, preserves conflicting
values with provenance, and never turns history into current-page actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any

from shopping_grpo.environment.observation import (
    OBSERVATION_VERSION,
    StructuredObservationError,
    render_structured_observation,
)
from shopping_grpo.environment.product_id import is_product_id


_PRIVATE_FIELDS = {
    "answer",
    "goal",
    "gold",
    "reward",
    "reward_detail",
    "target_asin",
    "task_facts",
}
_PRODUCT_FIELDS = ("rank", "title", "brand", "category", "price", "key_attributes")


class MemoryBudgetError(RuntimeError):
    """The fixed memory header cannot fit the configured token budget."""


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize(value: Any) -> Any:
    """Return a stable, JSON-serializable representation of a public value."""
    if isinstance(value, Mapping):
        return {
            _text(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _text(value)


def _identity(value: Any) -> str:
    return json.dumps(
        _normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _reject_private_fields(value: Any, path: str = "observation_state") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _PRIVATE_FIELDS:
                raise StructuredObservationError(f"{path} contains forbidden field: {key}")
            _reject_private_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_fields(item, f"{path}[{index}]")


@dataclass(frozen=True, order=True)
class EvidenceSource:
    """Where a value was visible to the agent."""

    step_index: int
    tool_name: str
    page_type: str


@dataclass
class ObservedValue:
    """One distinct value plus every step that exposed it."""

    value: Any
    sources: list[EvidenceSource] = field(default_factory=list)

    def add_source(self, source: EvidenceSource) -> bool:
        if source in self.sources:
            return False
        self.sources.append(source)
        self.sources.sort()
        return True


@dataclass
class CandidateRecord:
    """All public evidence collected for one catalog product."""

    asin: str
    facts: dict[str, list[ObservedValue]] = field(default_factory=dict)
    first_seen_step: int | None = None
    last_seen_step: int | None = None

    def observe(self, name: str, value: Any, source: EvidenceSource) -> tuple[bool, bool]:
        """Record a value and return ``(new_value, new_source)``."""
        normalized = _normalize(value)
        if normalized in (None, "", [], {}):
            return False, False
        values = self.facts.setdefault(name, [])
        identity = _identity(normalized)
        for observed in values:
            if _identity(observed.value) == identity:
                return False, observed.add_source(source)
        values.append(ObservedValue(value=normalized, sources=[source]))
        values.sort(key=lambda item: _identity(item.value))
        return True, True


@dataclass(frozen=True)
class MemoryUpdate:
    """Auditable summary of one observation update."""

    step_index: int
    page_type: str
    candidates_seen: int
    candidates_created: int
    values_added: int
    sources_added: int


@dataclass(frozen=True)
class MemoryView:
    """One budgeted prompt view and its keep/drop decisions."""

    content: str
    token_budget: int
    used_tokens: int
    selected_asins: tuple[str, ...]
    dropped_asins: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        content = value["content"]
        value["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        value["selected_asins"] = list(self.selected_asins)
        value["dropped_asins"] = list(self.dropped_asins)
        return value


@dataclass
class CandidateMemory:
    """Provenance-preserving memory over public shopping observations."""

    requirement: str
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    updates: list[MemoryUpdate] = field(default_factory=list)

    def observe(
        self, state: Mapping[str, Any], *, step_index: int, tool_name: str
    ) -> MemoryUpdate:
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(state, Mapping):
            raise StructuredObservationError("observation_state must be an object")
        _reject_private_fields(state)
        if state.get("observation_version") != OBSERVATION_VERSION:
            raise StructuredObservationError("unsupported observation_state version")

        # The environment renderer remains the source-of-truth schema validator.
        render_structured_observation(state)
        page_type = str(state.get("page_type") or "unknown")
        source = EvidenceSource(
            step_index=int(step_index),
            tool_name=_text(tool_name) or "unknown",
            page_type=page_type,
        )
        products = self._products_from_state(state, page_type)
        created = values_added = sources_added = 0
        for product, extra_facts in products:
            asin = _text(product.get("asin"))
            if not is_product_id(asin):
                raise StructuredObservationError(f"invalid product ASIN: {asin!r}")
            record = self.candidates.get(asin)
            if record is None:
                record = CandidateRecord(asin=asin)
                self.candidates[asin] = record
                created += 1
            if record.first_seen_step is None:
                record.first_seen_step = int(step_index)
            record.last_seen_step = max(record.last_seen_step or 0, int(step_index))
            facts = {name: product.get(name) for name in _PRODUCT_FIELDS}
            facts.update(extra_facts)
            for name, value in facts.items():
                new_value, new_source = record.observe(name, value, source)
                values_added += int(new_value)
                sources_added += int(new_source)

        update = MemoryUpdate(
            step_index=int(step_index),
            page_type=page_type,
            candidates_seen=len(products),
            candidates_created=created,
            values_added=values_added,
            sources_added=sources_added,
        )
        self.updates.append(update)
        return update

    @staticmethod
    def _products_from_state(
        state: Mapping[str, Any], page_type: str
    ) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
        if page_type == "search_results":
            return [(product, {}) for product in state.get("products") or []]
        if page_type not in {"product_detail", "information_subpage"}:
            return []
        product = state.get("product")
        if not isinstance(product, Mapping):
            raise StructuredObservationError("product page must contain a product object")
        extra: dict[str, Any] = {
            "price": state.get("selected_price", product.get("price")),
        }
        for name, value in sorted((state.get("selected_options") or {}).items()):
            extra[f"selected_option.{_text(name)}"] = value
        for name, value in sorted((state.get("available_options") or {}).items()):
            extra[f"available_option.{_text(name)}"] = value
        if page_type == "information_subpage":
            subpage = _text(state.get("subpage")) or "unknown"
            extra[f"subpage.{subpage}"] = state.get("content")
        return [(product, extra)]

    def snapshot(self) -> dict[str, Any]:
        """Return deterministic, machine-readable memory for trace auditing."""
        candidates = []
        for asin, record in sorted(self.candidates.items()):
            facts = {}
            for name, values in sorted(record.facts.items()):
                facts[name] = [
                    {
                        "value": observed.value,
                        "sources": [asdict(source) for source in observed.sources],
                    }
                    for observed in values
                ]
            candidates.append(
                {
                    "asin": asin,
                    "first_seen_step": record.first_seen_step,
                    "last_seen_step": record.last_seen_step,
                    "facts": facts,
                }
            )
        return {
            "requirement": _text(self.requirement),
            "candidates": candidates,
            "updates": [asdict(update) for update in self.updates],
        }

    def render(self) -> str:
        """Render evidence for a prompt without exposing historical actions."""
        lines = self._header_lines()
        for _, block in self._candidate_blocks():
            lines.extend(block)
        return "\n".join(lines)

    def render_view(self, *, token_budget: int, count_tokens) -> MemoryView:
        """Select whole candidate blocks under an exact model-token budget."""
        token_budget = int(token_budget)
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        lines = self._header_lines()
        header = "\n".join(lines)
        header_tokens = int(count_tokens(header))
        if header_tokens > token_budget:
            raise MemoryBudgetError(
                f"memory header uses {header_tokens} tokens, above budget {token_budget}"
            )

        blocks = self._candidate_blocks()
        low, high = 0, len(blocks)
        kept_count = 0
        content = header
        used_tokens = header_tokens
        # Strict-prefix selection is monotonic, so exact token counting only
        # needs O(log n) tokenizer requests instead of one request per item.
        while low <= high:
            middle = (low + high) // 2
            candidate_lines = [*lines]
            for _, block in blocks[:middle]:
                candidate_lines.extend(block)
            candidate = "\n".join(candidate_lines)
            candidate_tokens = int(count_tokens(candidate))
            if candidate_tokens <= token_budget:
                kept_count = middle
                content = candidate
                used_tokens = candidate_tokens
                low = middle + 1
            else:
                high = middle - 1
        selected = [asin for asin, _ in blocks[:kept_count]]
        dropped = [asin for asin, _ in blocks[kept_count:]]
        return MemoryView(
            content=content,
            token_budget=token_budget,
            used_tokens=used_tokens,
            selected_asins=tuple(selected),
            dropped_asins=tuple(dropped),
        )

    def _header_lines(self) -> list[str]:
        return [
            "[SHOPPING_CANDIDATE_MEMORY_V1]",
            "以下仅是历史比较证据，不是当前页面控件。",
            "历史候选标签和字段都不得直接用于工具参数；下一步只能使用 CURRENT_ACTION_STATE 中列出的目标。",
        ]

    def _candidate_blocks(self) -> list[tuple[str, list[str]]]:
        """Rank verified and recent candidates before cheap search-only leads."""
        candidates = self.snapshot()["candidates"]
        stable_order = sorted(
            candidates,
            key=lambda candidate: (
                candidate["first_seen_step"] if candidate["first_seen_step"] is not None else 10**9,
                candidate["asin"],
            ),
        )
        labels = {
            candidate["asin"]: f"H{index:02d}"
            for index, candidate in enumerate(stable_order, start=1)
        }

        def priority(candidate):
            facts = candidate["facts"]
            verified = sum(
                name.startswith(("selected_option.", "subpage.")) for name in facts
            )
            ranks = [
                value["value"]
                for value in facts.get("rank", [])
                if isinstance(value["value"], int)
            ]
            best_rank = min(ranks, default=10**9)
            return (
                -verified,
                -(candidate["last_seen_step"] or 0),
                best_rank,
                -len(facts),
                candidate["asin"],
            )

        blocks = []
        for candidate in sorted(candidates, key=priority):
            lines = [
                f"历史候选 {labels[candidate['asin']]} "
                f"(seen {candidate['first_seen_step']}-{candidate['last_seen_step']})"
            ]
            for name, values in candidate["facts"].items():
                rendered_values = []
                for observed in values:
                    source_text = ",".join(
                        f"s{source['step_index']}:{source['tool_name']}:{source['page_type']}"
                        for source in observed["sources"]
                    )
                    value_text = json.dumps(
                        observed["value"], ensure_ascii=False, sort_keys=True
                    )
                    rendered_values.append(f"{value_text} [{source_text}]")
                lines.append(f"- {name}: " + " | ".join(rendered_values))
            blocks.append((candidate["asin"], lines))
        return blocks
