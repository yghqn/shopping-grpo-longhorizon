"""Deterministic Rubric candidates and code-owned Flash materialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.evaluation.contracts import (
    RUBRIC_SCHEMA_VERSION,
    ContractValidationError,
    validate_curator_response,
    validate_rubric_bundle,
)

TASK_FACTS_VERSION = "shopping-task-facts-v1"
RUBRIC_CANDIDATE_VERSION = "shopping-rubric-candidates-v1"
RUBRIC_EXTRACTOR_VERSION = "shopping-rubric-extractor-v6"

_NUMBER = r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?"
_CURRENCY = r"(?:元|块)"
_UPPER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:控制在?|要|得|需|需要)?\s*"
    rf"(?:不超过|不要超过|别超|不能超过|不得超过|不可超过|"
    rf"不高于|最高|至多|封顶(?:在)?)\s*"
    rf"{_NUMBER}\s*{_CURRENCY}?",
)
_UPPER_SUFFIX = re.compile(
    rf"{_NUMBER}\s*{_CURRENCY}?\s*(?:以内|以下|之内)"
)
_LOWER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:不低于|至少|最低|不少于|高于|超过)\s*"
    rf"{_NUMBER}\s*{_CURRENCY}?",
)
_LOWER_SUFFIX = re.compile(
    rf"{_NUMBER}\s*{_CURRENCY}?\s*(?:以上|起)"
)
_PLUS_LOWER = re.compile(rf"{_NUMBER}\s*{_CURRENCY}?\s*\+")
_RANGE = re.compile(
    rf"(?:预算|价格)?(?:控制)?在?\s*{_NUMBER}\s*{_CURRENCY}?\s*"
    rf"(?:-|~|～|至|到)\s*{_NUMBER}\s*{_CURRENCY}?(?:之间)?"
)
_QUERY_FILLER = re.compile(
    r"(?:请帮我|帮我|请|推荐|寻找|找一?下?|求|想选个?|想要|希望|需要|"
    r"配备|能作为|作为|用来|用于|物品|商品|一款|这款|该款|这件|这个|"
    r"款的|比较|看上去)"
)
_QUERY_SEPARATOR = re.compile(r"[\s，,。；;：:！？!?、（）()]+")
_AROUND = re.compile(
    rf"(?:预算|价格)?(?:控制)?在?\s*{_NUMBER}\s*{_CURRENCY}?\s*"
    rf"(?:左右|上下)"
)
_WAN_SHORTHAND = r"(\d+(?:\.\d+)?)\s*万\s*(\d+(?:\.\d+)?)\s*(千)?"
_WAN_UPPER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:控制在?|要|得|需|需要)?\s*"
    rf"(?:不超过|不要超过|别超|不能超过|不得超过|不可超过|"
    rf"不高于|最高|至多|封顶(?:在)?)\s*"
    rf"{_WAN_SHORTHAND}\s*{_CURRENCY}?"
)
_WAN_UPPER_SUFFIX = re.compile(
    rf"{_WAN_SHORTHAND}\s*{_CURRENCY}?\s*(?:以内|以下|之内)"
)
_PRICE_CUE = re.compile(r"预算|价格")
_PRICE_CURRENCY = re.compile(_CURRENCY)
_INCOMPLETE_QUERY_FRAGMENT = re.compile(
    r"(?:直径约?|内长|长度|宽度|高度|尺寸|规格)$"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in result:
            result.append(text)
    return result


def _scaled(number: str, unit: str | None) -> float:
    value = float(number)
    normalized = str(unit or "").casefold()
    if normalized == "万":
        value *= 10000
    elif normalized in {"千", "k"}:
        value *= 1000
    return value


def _scaled_wan_shorthand(
    major: str,
    tail: str,
    tail_unit: str | None,
) -> float:
    """Parse colloquial forms such as ``1万2`` and ``1万2千`` as 12,000."""

    value = float(major) * 10000
    remainder = float(tail)
    if tail_unit == "千" or remainder < 10:
        remainder *= 1000
    return value + remainder


def _span(query: str, start: int, end: int) -> dict:
    return {"text": query[start:end], "start": start, "end": end}


def _exact_spans(query: str, value: str) -> list[dict]:
    if not value:
        return []
    return [
        _span(query, match.start(), match.end())
        for match in re.finditer(re.escape(value), query, flags=re.IGNORECASE)
    ]


def _option_query_spans(query: str, value: str) -> list[dict]:
    """Recover exact Query evidence from decorated simulator option labels."""

    spans = _exact_spans(query, value)
    chunks = re.findall(r"[0-9A-Za-z\u4e00-\u9fff-]{2,}", str(value))
    for chunk in chunks:
        if re.fullmatch(r"\d+个装", chunk):
            continue
        if re.fullmatch(r"[宽高长厚]\d", chunk):
            continue
        for span in _exact_spans(query, chunk):
            if span not in spans:
                spans.append(span)
    ignored_option_phrases = {
        "一个",
        "一片",
        "一对",
        "两个",
        "产品",
        "商品",
        "尺寸",
        "款式",
        "用于",
        "直径",
        "规格",
        "适用",
        "选择",
        "颜色",
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", str(value)):
        matched = False
        for width in range(min(6, len(chunk)), 1, -1):
            for start in range(len(chunk) - width + 1):
                phrase = chunk[start : start + width]
                if phrase in ignored_option_phrases:
                    continue
                normalized_phrase = _normalized_constraint_text(phrase)
                if any(
                    normalized_phrase
                    and normalized_phrase
                    in _normalized_constraint_text(span.get("text"))
                    for span in spans
                ):
                    continue
                phrase_spans = _exact_spans(query, phrase)
                if not phrase_spans:
                    continue
                for span in phrase_spans:
                    if span not in spans:
                        spans.append(span)
                matched = True
            if matched:
                break
    if re.search(r"(?:^|\D)2\s*个装", str(value)):
        for phrase in ("一对", "两个", "2个"):
            for span in _exact_spans(query, phrase):
                if span not in spans:
                    spans.append(span)

    unit_names = {
        "mm": ("mm", "毫米"),
        "cm": ("cm", "厘米"),
        "m": ("m", "米"),
    }
    for number, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m)(?![A-Za-z])",
        str(value),
        flags=re.IGNORECASE,
    ):
        for unit_name in unit_names[unit.lower()]:
            pattern = re.compile(
                rf"{re.escape(number)}\s*{re.escape(unit_name)}",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(query):
                span = _span(query, match.start(), match.end())
                if span not in spans:
                    spans.append(span)
    for low, high, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m)(?![A-Za-z])",
        str(value),
        flags=re.IGNORECASE,
    ):
        unit_alternatives = "|".join(
            re.escape(name) for name in unit_names[unit.lower()]
        )
        pattern = re.compile(
            rf"{re.escape(low)}\s*(?:-|~|～|至|到)\s*"
            rf"{re.escape(high)}\s*(?:{unit_alternatives})",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(query):
            span = _span(query, match.start(), match.end())
            if span not in spans:
                spans.append(span)
    return sorted(spans, key=lambda item: (item["start"], item["end"]))


def _uncovered_query_phrase_candidates(
    query: str,
    existing_candidates: list[Mapping],
    *,
    product: Mapping,
) -> list[dict]:
    """Expose grounded Query fragments omitted by reward-feature metadata."""

    evidence_parts = [
        product.get("title"),
        product.get("category"),
        product.get("brand"),
        product.get("shop_name"),
    ]
    evidence_parts.extend(product.get("attributes") or [])
    options = product.get("customization_options")
    if isinstance(options, Mapping):
        for values in options.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, Mapping):
                    evidence_parts.append(value.get("value"))
                else:
                    evidence_parts.append(value)
    evidence = _normalized_constraint_text(
        " ".join(str(value or "") for value in evidence_parts)
    )

    def clean_fragment(value: str) -> str:
        text = value.strip("的地得款个")
        text = re.sub(
            r"^(?:我是|我想要?|我想买|我想选|最近|能不能|给我|"
            r"想买个?|买个?|用的)+",
            "",
            text,
        )
        text = re.sub(r"(?:就行|即可)$", "", text)
        return text.strip("的地得款个")

    def grounded(value: str) -> bool:
        normalized = _normalized_constraint_text(value)
        if len(normalized) < 2:
            return False
        if normalized in evidence:
            return True
        units = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", normalized)
        compact = "".join(units)
        return any(
            compact[index : index + 2] in evidence
            for index in range(max(0, len(compact) - 1))
        )

    covered = [False] * len(query)

    def mark(start: int, end: int) -> None:
        for index in range(max(0, start), min(len(query), end)):
            covered[index] = True

    for candidate in existing_candidates:
        for span in candidate.get("query_spans") or []:
            if isinstance(span, Mapping):
                mark(int(span.get("start", 0)), int(span.get("end", 0)))
    for pattern in (_QUERY_FILLER, _QUERY_SEPARATOR):
        for match in pattern.finditer(query):
            mark(match.start(), match.end())

    fragments = []
    start = None
    for index in range(len(query) + 1):
        is_open = index < len(query) and not covered[index]
        if is_open and start is None:
            start = index
        if not is_open and start is not None:
            raw_text = query[start:index]
            text = clean_fragment(raw_text)
            text_start = query.find(text, start, index) if text else -1
            if (
                len(text) >= 2
                and text_start >= 0
                and not re.fullmatch(r"[0-9.]+", text)
                and not _INCOMPLETE_QUERY_FRAGMENT.search(text)
                and text
                not in {
                    "比如",
                    "那种",
                    "这种",
                    "适合",
                    "同时带有",
                    "颜色为",
                    "配置为",
                    "预算在",
                    "价格预算在",
                    "材质为",
                    "采用",
                    "结构",
                    "成分",
                    "品牌",
                }
                and grounded(text)
            ):
                fragments.append(
                    {
                        "constraint_type": "query_phrase",
                        "description_hint": f"商品应满足明确要求：{text}",
                        "field_path": "product.visible_evidence",
                        "operator": "semantically_supports",
                        "expected_value": text,
                        "hardness_hint": "hard",
                        "hardness_source": "explicit_query_phrase_rule",
                        "query_spans": [
                            _span(query, text_start, text_start + len(text))
                        ],
                        "data_sources": ["query"],
                        "selection_guidance": (
                            "仅当该原文片段表达尚未被更具体候选覆盖的独立商品"
                            "要求时选择；意图套话、使用场景连接词和已被 option、"
                            "品类、功能或价格候选覆盖的内容不得重复选择。"
                        ),
                    }
                )
            start = None
    return fragments


def extract_price_candidates(query: str) -> list[dict]:
    """Extract explicit price semantics for Rubric candidates, not Reward."""

    query = str(query or "")
    candidates = []
    occupied = []

    def overlaps(match: re.Match) -> bool:
        return any(
            match.start() < previous_end and previous_start < match.end()
            for previous_start, previous_end in occupied
        )

    def has_price_context(match: re.Match) -> bool:
        text = match.group(0)
        return bool(
            _PRICE_CUE.search(text) or _PRICE_CURRENCY.search(text)
        )

    for pattern in (_WAN_UPPER_PREFIX, _WAN_UPPER_SUFFIX):
        for match in pattern.finditer(query):
            if overlaps(match) or not has_price_context(match):
                continue
            value = _scaled_wan_shorthand(
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if value <= 0:
                continue
            occupied.append((match.start(), match.end()))
            candidates.append(
                {
                    "constraint_type": "budget_upper",
                    "description_hint": f"价格不应超过{value:g}元",
                    "field_path": "purchase.price",
                    "operator": "lte",
                    "expected_value": {
                        "value": value,
                        "currency": "CNY",
                    },
                    "hardness_hint": "hard",
                    "hardness_source": "explicit_upper_budget_rule",
                    "query_spans": [
                        _span(query, match.start(), match.end())
                    ],
                    "data_sources": ["query"],
                }
            )
    for match in _RANGE.finditer(query):
        if not has_price_context(match):
            continue
        low = _scaled(match.group(1), match.group(2))
        high = _scaled(match.group(3), match.group(4))
        if low <= 0 or high < low:
            continue
        occupied.append((match.start(), match.end()))
        candidates.append(
            {
                "constraint_type": "price_range",
                "description_hint": (
                    f"价格应在{low:g}元到{high:g}元之间"
                ),
                "field_path": "purchase.price",
                "operator": "between",
                "expected_value": {"min": low, "max": high, "currency": "CNY"},
                "hardness_hint": "hard",
                "hardness_source": "explicit_price_range_rule",
                "query_spans": [_span(query, match.start(), match.end())],
                "data_sources": ["query"],
            }
        )
    for pattern, operator, constraint_type, hardness, source in (
        (
            _UPPER_PREFIX,
            "lte",
            "budget_upper",
            "hard",
            "explicit_upper_budget_rule",
        ),
        (
            _UPPER_SUFFIX,
            "lte",
            "budget_upper",
            "hard",
            "explicit_upper_budget_rule",
        ),
        (
            _LOWER_PREFIX,
            "gte",
            "price_lower",
            "hard",
            "explicit_lower_price_rule",
        ),
        (
            _LOWER_SUFFIX,
            "gte",
            "price_lower",
            "hard",
            "explicit_lower_price_rule",
        ),
        (
            _PLUS_LOWER,
            "gte",
            "price_lower",
            "hard",
            "explicit_plus_lower_rule",
        ),
        (
            _AROUND,
            "approximately",
            "price_preference",
            "soft",
            "approximate_price_preference_rule",
        ),
    ):
        for match in pattern.finditer(query):
            if overlaps(match) or not has_price_context(match):
                continue
            occupied.append((match.start(), match.end()))
            value = _scaled(match.group(1), match.group(2))
            if operator == "lte":
                description = f"价格不应超过{value:g}元"
            elif operator == "gte":
                description = f"价格不应低于{value:g}元"
            else:
                description = f"价格倾向于{value:g}元左右"
            candidates.append(
                {
                    "constraint_type": constraint_type,
                    "description_hint": description,
                    "field_path": "purchase.price",
                    "operator": operator,
                    "expected_value": {
                        "value": value,
                        "currency": "CNY",
                    },
                    "hardness_hint": hardness,
                    "hardness_source": source,
                    "query_spans": [_span(query, match.start(), match.end())],
                    "data_sources": ["query"],
                }
            )
    return sorted(
        candidates,
        key=lambda item: item["query_spans"][0]["start"],
    )


def build_task_facts(
    *,
    task_id: int,
    query: str,
    target_product: Mapping,
    instruction_record: Mapping,
    reward_goal: Mapping | None = None,
) -> dict:
    """Build a private task snapshot without exposing it to the Actor/Judge."""

    product = target_product if isinstance(target_product, Mapping) else {}
    instruction = (
        instruction_record if isinstance(instruction_record, Mapping) else {}
    )
    goal = reward_goal if isinstance(reward_goal, Mapping) else {}
    target = {
        "asin": product.get("asin"),
        "category": product.get("category"),
        "title": product.get("title"),
        "brand": product.get("brand"),
        "shop_name": product.get("shop_name"),
        "pricing": deepcopy(product.get("pricing")),
        "attributes": deepcopy(product.get("attribute")),
        "customization_options": deepcopy(
            product.get("customization_options")
        ),
    }
    instruction_view = {
        "instruction": str(query or instruction.get("instruction") or ""),
        "attributes": _clean_list(instruction.get("attributes")),
        "instruction_options": _clean_list(
            instruction.get("instruction_options")
        ),
    }
    reward_features = {
        key: deepcopy(goal.get(key))
        for key in (
            "category",
            "expected_brand",
            "expected_model",
            "expected_core_functions",
            "required_options_by_key",
            "unresolved_option_requirements",
            "price_upper",
            "reward_feature_version",
        )
        if key in goal
    }
    payload = {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": int(task_id),
        "query": instruction_view["instruction"],
        "instruction": instruction_view,
        "target_product": target,
        "reward_features": reward_features,
    }
    payload["task_data_hash"] = stable_hash(payload)
    payload["query_hash"] = stable_hash(payload["query"])
    return payload


def extract_rubric_candidates(task_facts: object) -> dict:
    """Extract a superset of constraints; Flash must decide which are requested."""

    if not isinstance(task_facts, Mapping):
        raise TypeError("task_facts must be an object")
    if task_facts.get("schema_version") != TASK_FACTS_VERSION:
        raise ValueError("unsupported task facts schema")
    query = str(task_facts.get("query") or "")
    instruction = task_facts.get("instruction")
    instruction = instruction if isinstance(instruction, Mapping) else {}
    product = task_facts.get("target_product")
    product = product if isinstance(product, Mapping) else {}
    reward = task_facts.get("reward_features")
    reward = reward if isinstance(reward, Mapping) else {}
    raw_candidates = []

    category = reward.get("category") or product.get("category")
    if category:
        leaf = str(category).split("›")[-1].strip()
        raw_candidates.append(
            {
                "constraint_type": "category",
                "description_hint": f"商品品类应为{leaf}",
                "field_path": "product.category",
                "operator": "in_category",
                "expected_value": str(category),
                "hardness_hint": "hard",
                "hardness_source": "explicit_category_rule",
                "query_spans": _exact_spans(query, leaf),
                "data_sources": ["task.target_product.category", "query"],
                "selection_guidance": (
                    "Query 明确请求该商品类型或无歧义同义品类时应选。"
                ),
            }
        )

    for value in _clean_list(reward.get("expected_brand")):
        raw_candidates.append(
            {
                "constraint_type": "brand",
                "description_hint": f"品牌应为{value}",
                "field_path": "product.brand",
                "operator": "eq",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_brand_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["reward_features.expected_brand", "query"],
                "selection_guidance": (
                    "仅当 Query 要求所购商品本身属于该品牌时选择；兼容/适用于"
                    "某品牌、或与品牌值同名的商品品类词都不是品牌要求。"
                ),
            }
        )
    for value in _clean_list(reward.get("expected_model")):
        raw_candidates.append(
            {
                "constraint_type": "model",
                "description_hint": f"型号应为{value}",
                "field_path": "product.model",
                "operator": "eq",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_model_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["reward_features.expected_model", "query"],
                "selection_guidance": (
                    "仅当 Query 将该值明确作为型号时选择；若该值只是材质牌号且"
                    "已有更完整的材质或功能候选，应选择后者。"
                ),
            }
        )

    attributes = _clean_list(
        reward.get("expected_core_functions")
        or instruction.get("attributes")
    )
    for value in attributes:
        raw_candidates.append(
            {
                "constraint_type": "core_function",
                "description_hint": f"商品应支持{value}",
                "field_path": "product.attributes",
                "operator": "contains",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_core_function_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["instruction.attributes", "query"],
                "selection_guidance": (
                    "Query 明确表达相同功能、适用人群、使用场景或无歧义同义"
                    "要求时应选；同一含义有多个候选时只选最具体者。"
                ),
            }
        )

    required_options = reward.get("required_options_by_key")
    if isinstance(required_options, Mapping):
        for canonical_axis, requirement in required_options.items():
            if isinstance(requirement, Mapping):
                option_values = [requirement.get("value")]
                source_axis = (
                    requirement.get("source_axis") or canonical_axis
                )
            elif isinstance(requirement, list):
                option_values = requirement
                source_axis = canonical_axis
            else:
                option_values = [requirement]
                source_axis = canonical_axis
            for option_value in _clean_list(option_values):
                raw_candidates.append(
                    {
                        "constraint_type": "option",
                        "description_hint": (
                            f"{source_axis}应选择{option_value}"
                        ),
                        "field_path": (
                            f"purchase.options.{canonical_axis}"
                        ),
                        "operator": "eq",
                        "expected_value": option_value,
                        "hardness_hint": "hard",
                        "hardness_source": "explicit_option_rule",
                        "query_spans": _option_query_spans(query, option_value),
                        "data_sources": [
                            "instruction.instruction_options",
                            "target_product.customization_options",
                            "query",
                        ],
                        "selection_guidance": (
                            "Query 明确要求该选项或其中组合规格时选择；如果值引入"
                            "用户未要求的品牌、型号、数量或实质规格则拒绝。纯营销"
                            "标签可忽略，但其余内容必须能唯一承载 Query 的组合要求。"
                        ),
                    }
                )
    else:
        for option_value in _clean_list(
            instruction.get("instruction_options")
        ):
            raw_candidates.append(
                {
                    "constraint_type": "option",
                    "description_hint": f"规格应包含{option_value}",
                    "field_path": "purchase.options.unresolved",
                    "operator": "contains",
                    "expected_value": option_value,
                    "hardness_hint": "hard",
                    "hardness_source": "explicit_option_rule",
                    "query_spans": _option_query_spans(query, option_value),
                    "data_sources": [
                        "instruction.instruction_options",
                        "query",
                    ],
                }
            )

    raw_candidates.extend(extract_price_candidates(query))
    raw_candidates.extend(
        _uncovered_query_phrase_candidates(
            query,
            raw_candidates,
            product=product,
        )
    )
    candidates = []
    seen = set()
    for raw in raw_candidates:
        identity = stable_hash(
            {
                "constraint_type": raw.get("constraint_type"),
                "field_path": raw.get("field_path"),
                "operator": raw.get("operator"),
                "expected_value": raw.get("expected_value"),
            }
        )
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(
            {
                "candidate_id": f"c{len(candidates) + 1:04d}",
                **raw,
            }
        )
    return {
        "schema_version": RUBRIC_CANDIDATE_VERSION,
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "task_id": int(task_facts["task_id"]),
        "query": query,
        "task_data_hash": str(task_facts.get("task_data_hash") or ""),
        "query_hash": str(task_facts.get("query_hash") or ""),
        "candidates": candidates,
    }


def _spans_from_quote(query: str, quote: str) -> list[dict]:
    quote = str(quote or "").strip()
    if not quote:
        return []
    return _exact_spans(query, quote)


def _normalized_constraint_text(value: object) -> str:
    """Normalize labels only for deterministic duplicate suppression."""

    if not isinstance(value, str):
        return ""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


_GENERIC_OPTION_OVERLAP = {
    "一个",
    "一片",
    "一对",
    "两个",
    "产品",
    "商品",
    "安装",
    "尺寸",
    "款式",
    "用于",
    "直径",
    "规格",
    "适用",
    "选择",
    "颜色",
}


def _shares_specific_option_text(left: object, right: object) -> bool:
    left_text = _normalized_constraint_text(left)
    right_text = _normalized_constraint_text(right)
    if not left_text or not right_text:
        return False
    if len(left_text) <= len(right_text):
        shorter, longer = left_text, right_text
    else:
        shorter, longer = right_text, left_text
    for width in range(min(6, len(shorter)), 1, -1):
        for start in range(len(shorter) - width + 1):
            token = shorter[start : start + width]
            if token in _GENERIC_OPTION_OVERLAP:
                continue
            if token in longer:
                return True
    return False


def _drop_constraints_shadowed_by_options(
    selected_constraints: list[Mapping],
    *,
    candidates_by_id: Mapping[str, Mapping],
) -> list[Mapping]:
    """Keep a specific option instead of double-counting its generic label.

    Reward metadata can expose the same user requirement both as a generic
    ``core_function`` and as the exact purchasable ``option``.  The Curator is
    asked to select the most specific candidate, but this deterministic guard
    keeps one missed instruction from inflating the frozen Rubric.
    """

    selected_options = []
    for selected in selected_constraints:
        candidate = candidates_by_id[str(selected["candidate_id"])]
        if candidate.get("constraint_type") != "option":
            continue
        normalized = _normalized_constraint_text(candidate.get("expected_value"))
        if normalized:
            selected_options.append(normalized)

    filtered = []
    for selected in selected_constraints:
        candidate = candidates_by_id[str(selected["candidate_id"])]
        constraint_type = candidate.get("constraint_type")
        if constraint_type == "core_function":
            normalized = _normalized_constraint_text(
                candidate.get("expected_value")
            )
            if normalized and any(
                normalized in option for option in selected_options
            ):
                continue
        if constraint_type == "query_phrase" and any(
            _shares_specific_option_text(
                candidate.get("expected_value"),
                option,
            )
            for option in selected_options
        ):
            continue
        filtered.append(selected)
    return filtered


def materialize_rubric_bundle(
    *,
    task_facts: Mapping,
    candidates: Mapping,
    curator_response: Mapping,
    curator_model: str,
    curator_prompt_version: str,
    rubric_version: str,
) -> dict:
    """Freeze selected candidates while copying all semantic fields from code."""

    if task_facts.get("schema_version") != TASK_FACTS_VERSION:
        raise ContractValidationError("unsupported task facts schema")
    if candidates.get("schema_version") != RUBRIC_CANDIDATE_VERSION:
        raise ContractValidationError("unsupported Rubric candidate schema")
    if int(candidates.get("task_id")) != int(task_facts.get("task_id")):
        raise ContractValidationError(
            "candidate task_id does not match task facts"
        )
    if str(candidates.get("query")) != str(task_facts.get("query")):
        raise ContractValidationError("candidate Query does not match task facts")
    if str(candidates.get("task_data_hash")) != str(
        task_facts.get("task_data_hash")
    ):
        raise ContractValidationError(
            "candidate task_data_hash does not match task facts"
        )
    if str(candidates.get("query_hash")) != str(
        task_facts.get("query_hash")
    ):
        raise ContractValidationError(
            "candidate query_hash does not match task facts"
        )
    candidate_rows = candidates.get("candidates")
    if not isinstance(candidate_rows, list):
        raise ContractValidationError("candidates.candidates must be a list")
    by_id = {
        str(item.get("candidate_id")): item
        for item in candidate_rows
        if isinstance(item, Mapping)
    }
    response = validate_curator_response(
        curator_response,
        candidate_ids=by_id,
    )
    query = str(task_facts.get("query") or "")
    selected_constraints = _drop_constraints_shadowed_by_options(
        response["selected_constraints"],
        candidates_by_id=by_id,
    )
    rubrics = []
    for selected in selected_constraints:
        candidate = by_id[selected["candidate_id"]]
        hardness = selected["hardness"]
        quote_spans = _spans_from_quote(query, selected.get("query_quote", ""))
        rubrics.append(
            {
                "rubric_id": f"r{len(rubrics) + 1:04d}",
                "candidate_id": selected["candidate_id"],
                "constraint_type": candidate["constraint_type"],
                "description": selected["description"].strip(),
                "hardness": hardness,
                "hardness_source": candidate["hardness_source"],
                # Query spans are code-owned evidence. A curator paraphrase
                # falls back to the extractor's already validated spans.
                "query_spans": (
                    quote_spans
                    or deepcopy(candidate.get("query_spans") or [])
                ),
                "field_path": candidate["field_path"],
                "operator": candidate["operator"],
                "expected_value": deepcopy(candidate["expected_value"]),
                "data_sources": deepcopy(candidate["data_sources"]),
                "selection_reason": selected["selection_reason"].strip(),
            }
        )
    bundle = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_version": str(rubric_version),
        "task_id": int(task_facts["task_id"]),
        "query": query,
        "generation": {
            "extractor_version": str(candidates["extractor_version"]),
            "curator_model": str(curator_model),
            "curator_prompt_version": str(curator_prompt_version),
            "task_data_hash": str(task_facts["task_data_hash"]),
            "query_hash": str(task_facts["query_hash"]),
        },
        "rubrics": rubrics,
    }
    return validate_rubric_bundle(bundle)
