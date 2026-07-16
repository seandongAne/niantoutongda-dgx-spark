"""A1 synthetic narration benchmark: deterministic truth, scoring, and stop rules.

The benchmark deliberately keeps the oracle outside the model under test: narration
facts are generated from deterministic templates, while cloud/local audio models only
synthesise or extract those facts.  Model output remains evaluation evidence and never
becomes product ground truth.
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "a1-robustness-v1"

SYSTEM_PROMPT = (
    "你是搬家助手的旁白解析器。输入是用户拍摄房间时的口述旁白语音。"
    "抽取旁白中提到的每一件物品，输出 JSON 数组，每项格式："
    '{"label_zh": 中文名, "label_en": 英文检测短语(1-3词), '
    '"owner": 所属人或null, "source_location": 当前位置或null, '
    '"target_location": 搬运去向或null, "pack_group": 同包分组要求或null, '
    '"attributes": {"color": 颜色或null}}。只输出 JSON 数组，不要任何解释。'
)

REQUIRED_OUTPUT_KEYS = {
    "label_zh",
    "label_en",
    "owner",
    "source_location",
    "target_location",
    "pack_group",
    "attributes",
}
SLOT_KEYS = ("owner", "source_location", "target_location", "pack_group", "color")

CATALOG: tuple[dict[str, Any], ...] = (
    {"id": "water_bottle", "zh": "水杯", "en": "water bottle", "zh_aliases": ["水杯", "水壶"], "en_aliases": ["water bottle", "bottle", "tumbler"]},
    {"id": "table_lamp", "zh": "台灯", "en": "table lamp", "zh_aliases": ["台灯", "桌灯"], "en_aliases": ["table lamp", "desk lamp", "lamp"]},
    {"id": "luggage", "zh": "行李箱", "en": "luggage", "zh_aliases": ["行李箱", "旅行箱"], "en_aliases": ["luggage", "suitcase"]},
    {"id": "security_camera", "zh": "摄像头", "en": "security camera", "zh_aliases": ["摄像头", "监控摄像头"], "en_aliases": ["security camera", "camera", "baby monitor"]},
    {"id": "stuffed_animal", "zh": "毛绒玩具", "en": "stuffed animal", "zh_aliases": ["毛绒玩具", "玩偶"], "en_aliases": ["stuffed animal", "plush toy", "plushie"]},
    {"id": "book", "zh": "书", "en": "book", "zh_aliases": ["书", "图书"], "en_aliases": ["book"]},
    {"id": "laptop", "zh": "笔记本电脑", "en": "laptop", "zh_aliases": ["笔记本电脑", "电脑"], "en_aliases": ["laptop", "notebook computer"]},
    {"id": "charging_cable", "zh": "充电线", "en": "charging cable", "zh_aliases": ["充电线", "数据线"], "en_aliases": ["charging cable", "charger cable", "usb cable"]},
    {"id": "alarm_clock", "zh": "闹钟", "en": "alarm clock", "zh_aliases": ["闹钟", "时钟"], "en_aliases": ["alarm clock", "clock"]},
    {"id": "storage_box", "zh": "收纳箱", "en": "storage box", "zh_aliases": ["收纳箱", "储物箱"], "en_aliases": ["storage box", "storage bin"]},
    {"id": "pillow", "zh": "枕头", "en": "pillow", "zh_aliases": ["枕头", "靠枕"], "en_aliases": ["pillow", "cushion"]},
    {"id": "headphones", "zh": "耳机", "en": "headphones", "zh_aliases": ["耳机"], "en_aliases": ["headphones", "headset"]},
)

COLORS = (
    ("白色", "white"),
    ("黑色", "black"),
    ("蓝色", "blue"),
    ("红色", "red"),
    ("粉色", "pink"),
    ("绿色", "green"),
    ("黄色", "yellow"),
    ("棕色", "brown"),
)
OWNERS = ("小明", "妈妈", "爸爸", "姐姐")
SOURCES = ("旧卧室书桌上", "旧卧室床头柜上", "旧卧室衣柜旁", "旧卧室书架上")
TARGETS = ("新家卧室书桌上", "新家卧室床头柜上", "新家卧室衣柜旁", "新家卧室书架上")
PACK_GROUPS = ("学习用品", "睡前用品", "充电用品", "随身用品")
STYLES = ("direct", "reordered", "conversational")

CONDITIONS: dict[str, dict[str, Any]] = {
    "clean": {"kind": "clean"},
    "noise20": {"kind": "noise", "snr_db": 20.0},
    "noise10": {"kind": "noise", "snr_db": 10.0},
    "speed090": {"kind": "speed", "ratio": 0.9},
    "codec32": {"kind": "codec", "bitrate": "32k"},
}

_COLOR_ALIASES = {
    "白": "白色", "白色": "白色", "white": "白色",
    "黑": "黑色", "黑色": "黑色", "black": "黑色",
    "蓝": "蓝色", "蓝色": "蓝色", "blue": "蓝色",
    "红": "红色", "红色": "红色", "red": "红色",
    "粉": "粉色", "粉色": "粉色", "pink": "粉色",
    "绿": "绿色", "绿色": "绿色", "green": "绿色",
    "黄": "黄色", "黄色": "黄色", "yellow": "黄色",
    "棕": "棕色", "棕色": "棕色", "brown": "棕色",
}


class BenchmarkInputError(ValueError):
    """Raised when a benchmark plan or prediction violates the frozen format."""


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | int]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise BenchmarkInputError("Wilson counts must satisfy 0 <= successes <= total")
    if total == 0:
        return {"successes": successes, "total": total, "rate": 0.0, "low": 0.0, "high": 1.0, "half_width": 0.5}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": round(proportion, 6),
        "low": round(max(0.0, center - margin), 6),
        "high": round(min(1.0, center + margin), 6),
        "half_width": round(margin, 6),
    }


def required_wilson_trials(target_half_width: float, z: float = 1.96) -> int:
    """Worst-case per-stratum sample size required for the requested precision."""
    if not 0.0 < target_half_width < 0.5:
        raise BenchmarkInputError("target_half_width must be between 0 and 0.5")
    for total in range(1, 100_001):
        successes = total // 2
        if float(wilson_interval(successes, total, z)["half_width"]) <= target_half_width:
            return total
    raise BenchmarkInputError("target_half_width requires more than 100000 trials")


def build_plan(
    *,
    seed: int,
    base_cases: int | None,
    condition_ids: Sequence[str],
    voices: Sequence[str],
    target_half_width: float = 0.08,
    minimum_per_condition: int = 100,
    maximum_observations: int = 2000,
) -> dict[str, Any]:
    """Build a balanced deterministic benchmark plan.

    ``base_cases=None`` selects the statistically required size, capped by
    ``maximum_observations``.  A positive explicit count is used for calibration.
    """
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise BenchmarkInputError("condition_ids must be non-empty and unique")
    unknown = [condition for condition in condition_ids if condition not in CONDITIONS]
    if unknown:
        raise BenchmarkInputError(f"unknown conditions: {unknown}")
    if not voices:
        raise BenchmarkInputError("at least one TTS voice is required")
    required = max(minimum_per_condition, required_wilson_trials(target_half_width))
    cap_per_condition = maximum_observations // len(condition_ids)
    if cap_per_condition < 1:
        raise BenchmarkInputError("maximum_observations is smaller than condition count")
    selected = min(required, cap_per_condition) if base_cases is None else base_cases
    if selected < 1:
        raise BenchmarkInputError("base_cases must be positive")
    if selected * len(condition_ids) > maximum_observations:
        raise BenchmarkInputError("requested plan exceeds maximum_observations")

    cases = generate_cases(selected, seed=seed, voices=voices)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "oracle": "deterministic_templates",
        "privacy": "synthetic_only_no_household_media",
        "system_prompt": SYSTEM_PROMPT,
        "condition_ids": list(condition_ids),
        "conditions": {condition: CONDITIONS[condition] for condition in condition_ids},
        "voices": list(voices),
        "stopping_rule": {
            "metric": "slot_accuracy",
            "confidence": 0.95,
            "target_half_width": target_half_width,
            "minimum_per_condition": minimum_per_condition,
            "worst_case_required_per_condition": required,
            "maximum_observations": maximum_observations,
            "target_reachable_in_plan": selected >= required,
        },
        "base_case_count": selected,
        "observation_count_per_backend": selected * len(condition_ids),
        "cases": cases,
    }


def generate_cases(count: int, *, seed: int, voices: Sequence[str]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for index in range(count):
        style = STYLES[index % len(STYLES)]
        item_count = 1 + (index % 3)
        completeness = "full" if index % 2 == 0 else "partial"
        chosen = rng.sample(list(CATALOG), item_count)
        expected: list[dict[str, Any]] = []
        for item_index, item in enumerate(chosen):
            color_zh, color_en = COLORS[(index + item_index) % len(COLORS)]
            values: dict[str, Any] = {
                "canonical_id": item["id"],
                "label_zh": item["zh"],
                "label_en": item["en"],
                "owner": OWNERS[(index + item_index) % len(OWNERS)],
                "source_location": SOURCES[(index * 2 + item_index) % len(SOURCES)],
                "target_location": TARGETS[(index + item_index * 2) % len(TARGETS)],
                "pack_group": PACK_GROUPS[(index + item_index) % len(PACK_GROUPS)],
                "color": color_zh,
                "color_en": color_en,
            }
            if completeness == "partial":
                omitted = ("owner", "target_location", "pack_group")[(index + item_index) % 3]
                values[omitted] = None
            expected.append(values)
        narration = _render_narration(expected, style)
        cases.append({
            "case_id": f"case-{index + 1:04d}",
            "style": style,
            "item_count": item_count,
            "completeness": completeness,
            "voice": voices[index % len(voices)],
            "tts_instruction": "自然口语，像边拍摄房间边介绍，语速正常，吐字清楚",
            "narration": narration,
            "expected": [{key: value for key, value in item.items() if key != "color_en"} for item in expected],
        })
    return cases


def _render_narration(items: Sequence[Mapping[str, Any]], style: str) -> str:
    fragments: list[str] = []
    for item in items:
        label = f"{item['color']}{item['label_zh']}"
        owner = f"这是{item['owner']}的" if item["owner"] else "这是"
        source = f"现在放在{item['source_location']}"
        target = f"搬到新家后放到{item['target_location']}" if item["target_location"] else ""
        group = f"打包时归到{item['pack_group']}" if item["pack_group"] else ""
        parts = [part for part in (target, owner + label, source, group) if part]
        if style == "direct":
            parts = [owner + label, source, target, group]
        elif style == "conversational":
            parts = ["嗯，先看这个，" + owner + label, source, target, group]
        fragments.append("，".join(part for part in parts if part) + "。")
    return "接下来，".join(fragments) if style == "reordered" else "".join(fragments)


def parse_prediction(raw_text: str) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Parse a model response and report protocol-level schema validity."""
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline >= 0 else text.strip("`")
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return [], False, "missing_json_array"
    extra_text = bool(text[:start].strip() or text[end + 1:].strip())
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return [], False, f"invalid_json:{exc.msg}"
    if not isinstance(parsed, list):
        return [], False, "top_level_not_array"
    schema_valid = not extra_text
    for value in parsed:
        if not isinstance(value, dict) or not REQUIRED_OUTPUT_KEYS.issubset(value):
            schema_valid = False
            continue
        if not isinstance(value.get("attributes"), dict) or "color" not in value["attributes"]:
            schema_valid = False
    return [value for value in parsed if isinstance(value, dict)], schema_valid, None if schema_valid else "schema_mismatch"


def score_case(case: Mapping[str, Any], raw_text: str) -> dict[str, Any]:
    predicted, schema_valid, parse_error = parse_prediction(raw_text)
    expected_by_id = {item["canonical_id"]: item for item in case["expected"]}
    predicted_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    unrecognized = 0
    for item in predicted:
        canonical_id = canonical_id_for(item)
        if canonical_id is None:
            unrecognized += 1
        elif canonical_id in predicted_by_id:
            duplicate_ids.append(canonical_id)
        else:
            predicted_by_id[canonical_id] = item

    matched_ids = sorted(set(expected_by_id) & set(predicted_by_id))
    slot_correct = 0
    slot_total = 0
    field_scores: dict[str, dict[str, int]] = {key: {"correct": 0, "total": 0} for key in SLOT_KEYS}
    for canonical_id, expected in expected_by_id.items():
        if canonical_id not in predicted_by_id:
            slot_total += len(SLOT_KEYS)
            for field in SLOT_KEYS:
                field_scores[field]["total"] += 1
            continue
        actual = predicted_by_id[canonical_id]
        for field in SLOT_KEYS:
            expected_value = expected[field]
            actual_value = actual.get("attributes", {}).get("color") if field == "color" else actual.get(field)
            is_correct = normalize_slot(field, actual_value) == normalize_slot(field, expected_value)
            slot_total += 1
            field_scores[field]["total"] += 1
            if is_correct:
                slot_correct += 1
                field_scores[field]["correct"] += 1

    expected_count = len(expected_by_id)
    predicted_count = len(predicted)
    item_tp = len(matched_ids)
    false_positive_items = max(0, predicted_count - item_tp)
    exact_case = (
        schema_valid
        and item_tp == expected_count
        and predicted_count == expected_count
        and not duplicate_ids
        and slot_correct == slot_total == expected_count * len(SLOT_KEYS)
    )
    return {
        "case_id": case["case_id"],
        "style": case["style"],
        "voice": case["voice"],
        "completeness": case["completeness"],
        "schema_valid": schema_valid,
        "parse_error": parse_error,
        "exact_case": exact_case,
        "expected_items": expected_count,
        "predicted_items": predicted_count,
        "item_true_positives": item_tp,
        "false_positive_items": false_positive_items,
        "unrecognized_items": unrecognized,
        "duplicate_canonical_ids": duplicate_ids,
        "slot_correct": slot_correct,
        "slot_total": slot_total,
        "field_scores": field_scores,
    }


def canonical_id_for(item: Mapping[str, Any]) -> str | None:
    zh = _normalize_label(item.get("label_zh"))
    en = _normalize_label(item.get("label_en"))
    color_words = set(_COLOR_ALIASES)
    en_words = " ".join(word for word in en.split() if word not in color_words)
    candidates: list[str] = []
    for catalog_item in CATALOG:
        zh_aliases = [_normalize_label(alias) for alias in catalog_item["zh_aliases"]]
        en_aliases = [_normalize_label(alias) for alias in catalog_item["en_aliases"]]
        if any(alias and (zh == alias or zh.endswith(alias)) for alias in zh_aliases):
            candidates.append(catalog_item["id"])
        elif any(alias and (en_words == alias or en_words.endswith(alias)) for alias in en_aliases):
            candidates.append(catalog_item["id"])
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def normalize_slot(field: str, value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if text in {"", "null", "none", "未知", "无", "未提及"}:
        return None
    if field == "color":
        compact = _normalize_label(text)
        return _COLOR_ALIASES.get(compact, compact)
    compact = re.sub(r"[\s，。,.、:：;；'\"“”‘’（）()\-]", "", text)
    compact = compact.replace("的", "")
    if field == "owner":
        compact = compact.removesuffix("所有")
    if field == "pack_group":
        compact = compact.removesuffix("一组").removesuffix("组")
    return compact or None


def _normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text).strip()


def aggregate_scores(records: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate scored case records per backend/condition and evaluate stop gates."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["backend"])].append(record)
    backends: dict[str, Any] = {}
    for backend, backend_records in sorted(grouped.items()):
        condition_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in backend_records:
            condition_groups[str(record["condition_id"])].append(record)
        conditions = {
            condition: _aggregate_group(group)
            for condition, group in sorted(condition_groups.items())
        }
        required = int(plan["stopping_rule"]["worst_case_required_per_condition"])
        target = float(plan["stopping_rule"]["target_half_width"])
        planned_conditions = list(plan["condition_ids"])
        missing_conditions = [condition for condition in planned_conditions if condition not in conditions]
        condition_ready = {
            condition: (
                conditions[condition]["cases"] >= required
                and conditions[condition]["slot_accuracy"]["half_width"] <= target
            ) if condition in conditions else False
            for condition in planned_conditions
        }
        minimum_observed = min(
            (conditions.get(condition, {}).get("cases", 0) for condition in planned_conditions),
            default=0,
        )
        backends[backend] = {
            **_aggregate_group(backend_records),
            "conditions": conditions,
            "coverage": _coverage(backend_records),
            "stopping": {
                "reached": bool(condition_ready) and all(condition_ready.values()),
                "condition_ready": condition_ready,
                "missing_conditions": missing_conditions,
                "required_per_condition": required,
                "minimum_observed_per_condition": minimum_observed,
                "additional_base_cases_needed": max(0, required - minimum_observed),
                "target_half_width": target,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "plan": {
            "seed": plan["seed"],
            "base_case_count": plan["base_case_count"],
            "condition_ids": plan["condition_ids"],
            "observation_count_per_backend": plan["observation_count_per_backend"],
            "stopping_rule": plan["stopping_rule"],
        },
        "backends": backends,
    }


def _aggregate_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases = len(records)
    schema_success = sum(bool(record["schema_valid"]) for record in records)
    exact_success = sum(bool(record["exact_case"]) for record in records)
    item_tp = sum(int(record["item_true_positives"]) for record in records)
    expected_items = sum(int(record["expected_items"]) for record in records)
    predicted_items = sum(int(record["predicted_items"]) for record in records)
    slot_correct = sum(int(record["slot_correct"]) for record in records)
    slot_total = sum(int(record["slot_total"]) for record in records)
    return {
        "cases": cases,
        "schema_valid_rate": wilson_interval(schema_success, cases),
        "exact_case_rate": wilson_interval(exact_success, cases),
        "item_recall": wilson_interval(item_tp, expected_items),
        "item_precision": wilson_interval(item_tp, max(predicted_items, item_tp)),
        "slot_accuracy": wilson_interval(slot_correct, slot_total),
        "parse_failures": sum(record.get("parse_error") is not None for record in records),
    }


def _coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for field in ("condition_id", "style", "voice", "completeness"):
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            counts[str(record[field])] += 1
        output[field] = dict(sorted(counts.items()))
    return output
