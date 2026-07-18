from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tools.risks import (
    DEFAULT_MIN_EVIDENCE_CONFIDENCE,
    RISK_DISCLAIMER_ZH,
    RULE_FACT_KEYS,
    RiskStatus,
    evaluate_risk_rule,
)


PROJ = Path(__file__).resolve().parent.parent.parent
CLOSURE_PATH = PROJ / "fixtures/hero_s1/technical_closure.json"
ITEMS_PATH = PROJ / "fixtures/hero_s1/items.json"
REGIONS_PATH = PROJ / "fixtures/hero_s1/regions.json"


def _fact(value: bool | None, confidence: float | None = 0.95, ref: str = "frame@00:01"):
    return {
        "value": value,
        "confidence": confidence,
        "evidence_refs": [ref] if ref else [],
    }


def _all_true(rule_id: str):
    return {
        key: _fact(True, 0.90 + idx * 0.01, f"frame:{key}")
        for idx, key in enumerate(RULE_FACT_KEYS[rule_id])
    }


def test_technical_closure_partitions_all_twenty_source_items_once():
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))
    source = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    source_ids = [item["canonical_id"] for item in source["items"]]

    assert len(source_ids) == 20
    assert closure["canonical_item_ids"] == source_ids

    assigned = [
        item_id
        for group in closure["life_groups"]
        for item_id in group["canonical_item_ids"]
    ] + closure["independent_pack_item_ids"]
    assert len(assigned) == 20
    assert len(set(assigned)) == 20
    assert set(assigned) == set(source_ids)
    assert [group["name_zh"] for group in closure["life_groups"]] == [
        "学习文具",
        "杯子饮品",
        "洗漱护理",
    ]


def test_technical_closure_freezes_gt_regions_and_pending_scale_slots():
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))
    regions = json.loads(REGIONS_PATH.read_text(encoding="utf-8"))
    known_region_ids = {entry["region_id"] for entry in regions["entries"]}
    ground_truth = closure["space_ground_truth"]

    assert ground_truth["usage"] == "ground_truth_acceptance_only"
    assert ground_truth["must_not_be_used_as_auto_spatial_producer_input"] is True
    assert len(ground_truth["target_region_ids"]) == 5
    assert set(ground_truth["target_region_ids"]) <= known_region_ids

    slots = closure["scale_measurement_inputs"]
    assert len(slots) == 3
    assert len({slot["slot_id"] for slot in slots}) == 3
    assert all(slot["status"] == "pending" for slot in slots)
    assert all(slot["value_cm"] is None for slot in slots)

    risk_contract = closure["risk_contract"]
    assert risk_contract["min_evidence_confidence"] == DEFAULT_MIN_EVIDENCE_CONFIDENCE
    assert risk_contract["disclaimer_zh"] == RISK_DISCLAIMER_ZH
    assert {
        rule["rule_id"]: tuple(rule["required_fact_keys"])
        for rule in risk_contract["rules"]
    } == RULE_FACT_KEYS


@pytest.mark.parametrize("rule_id", sorted(RULE_FACT_KEYS))
def test_each_rule_triggers_only_with_complete_sufficient_evidence(rule_id):
    result = evaluate_risk_rule(
        rule_id,
        _all_true(rule_id),
        subject_ids=["z", "a", "a"],
    )

    assert result.status == RiskStatus.TRIGGERED
    assert result.confidence >= DEFAULT_MIN_EVIDENCE_CONFIDENCE
    assert result.subject_ids == ["a", "z"]
    assert result.evidence_refs == sorted(
        f"frame:{key}" for key in RULE_FACT_KEYS[rule_id]
    )
    assert list(result.evidence) == list(RULE_FACT_KEYS[rule_id])
    assert result.disclaimer_zh == RISK_DISCLAIMER_ZH
    assert "不构成安全认证" in result.disclaimer_zh


@pytest.mark.parametrize("rule_id", sorted(RULE_FACT_KEYS))
def test_missing_fact_needs_user_and_never_triggers(rule_id):
    facts = _all_true(rule_id)
    missing = RULE_FACT_KEYS[rule_id][-1]
    facts.pop(missing)

    result = evaluate_risk_rule(rule_id, facts)

    assert result.status == RiskStatus.NEEDS_USER
    assert result.reason_codes == [f"MISSING_EVIDENCE:{missing}"]
    assert result.confidence == 0.0


def test_low_confidence_or_missing_reference_needs_user():
    low = _all_true("TRIP_HAZARD_IN_PATH")
    low["in_walk_path"] = _fact(True, 0.79, "frame:path")
    assert evaluate_risk_rule("TRIP_HAZARD_IN_PATH", low).reason_codes == [
        "LOW_CONFIDENCE:in_walk_path"
    ]

    no_ref = _all_true("POWER_IN_WET_ZONE")
    no_ref["wet_zone_present"] = _fact(True, 0.99, "")
    assert evaluate_risk_rule("POWER_IN_WET_ZONE", no_ref).reason_codes == [
        "MISSING_EVIDENCE_REF:wet_zone_present"
    ]


@pytest.mark.parametrize("rule_id", sorted(RULE_FACT_KEYS))
def test_sufficient_negative_fact_is_not_applicable(rule_id):
    facts = _all_true(rule_id)
    negative_key = RULE_FACT_KEYS[rule_id][0]
    facts[negative_key] = _fact(False, 0.96, f"frame:not-{negative_key}")

    result = evaluate_risk_rule(rule_id, facts)

    assert result.status == RiskStatus.NOT_APPLICABLE
    assert result.reason_codes == [f"NEGATED_TRIGGER_FACT:{negative_key}"]
    assert result.confidence == 0.96
    # 已有充分的非适用证据后不再消费后续事实，避免伪造额外依据。
    assert list(result.evidence) == [negative_key]


def test_rule_evaluation_is_deterministic_and_rejects_contract_drift():
    facts = _all_true("CHILD_SHARP_TOOL_REACH")
    first = evaluate_risk_rule("CHILD_SHARP_TOOL_REACH", facts)
    second = evaluate_risk_rule("CHILD_SHARP_TOOL_REACH", dict(reversed(list(facts.items()))))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")

    with pytest.raises(ValueError, match="unknown risk rule"):
        evaluate_risk_rule("UNKNOWN", {})
    with pytest.raises(ValueError, match="unexpected facts"):
        evaluate_risk_rule(
            "TRIP_HAZARD_IN_PATH",
            {**_all_true("TRIP_HAZARD_IN_PATH"), "hallucinated_safe": _fact(True)},
        )
