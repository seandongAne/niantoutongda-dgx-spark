#!/usr/bin/env python
"""独立风险提醒任务：技术 closure + 规则事实 → 可审计评估与指标。

输入 facts 是以 ``rule_id`` 为键的 JSON 对象，每个值只允许包含
``facts`` 与 ``subject_ids``。closure 中存在但输入缺失的规则仍会按空事实
调用评估器，稳定产出 ``NEEDS_USER``；未知规则、未知事实或 closure 与代码
合同漂移则直接失败，不生成任何看似安全的结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.risks import (  # noqa: E402
    RISK_DISCLAIMER_ZH,
    RULE_FACT_KEYS,
    RiskStatus,
    evaluate_risk_rule,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def _closure_contract(
    closure: object,
) -> tuple[str, list[str], float, str, str, bool, str]:
    if not isinstance(closure, dict):
        raise ValueError("closure root must be an object")
    closure_id = closure.get("closure_id")
    if not isinstance(closure_id, str) or not closure_id:
        raise ValueError("closure_id must be a non-empty string")
    contract = closure.get("risk_contract")
    if not isinstance(contract, dict):
        raise ValueError("closure.risk_contract must be an object")

    raw_rules = contract.get("rules")
    if not isinstance(raw_rules, list) or len(raw_rules) != 3:
        raise ValueError("closure must freeze exactly three risk rules")

    rule_order: list[str] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("each closure risk rule must be an object")
        rule_id = raw_rule.get("rule_id")
        fact_keys = raw_rule.get("required_fact_keys")
        if not isinstance(rule_id, str) or rule_id not in RULE_FACT_KEYS:
            raise ValueError(f"unknown risk rule in closure: {rule_id}")
        if rule_id in rule_order:
            raise ValueError(f"duplicate risk rule in closure: {rule_id}")
        if not isinstance(fact_keys, list) or tuple(fact_keys) != RULE_FACT_KEYS[rule_id]:
            raise ValueError(f"fact contract drift for {rule_id}")
        rule_order.append(rule_id)

    if set(rule_order) != set(RULE_FACT_KEYS):
        raise ValueError("closure risk rule set differs from evaluator rule set")

    threshold = contract.get("min_evidence_confidence")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("min_evidence_confidence must be numeric")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_evidence_confidence must be within [0, 1]")

    disclaimer = contract.get("disclaimer_zh")
    if disclaimer != RISK_DISCLAIMER_ZH:
        raise ValueError("risk disclaimer drift between closure and evaluator")

    scope_status = contract.get("scope_status")
    if scope_status not in {"ACTIVE", "DEFERRED"}:
        raise ValueError("risk_contract.scope_status must be ACTIVE or DEFERRED")
    blocking = contract.get("blocking")
    if not isinstance(blocking, bool):
        raise ValueError("risk_contract.blocking must be boolean")
    defer_reason = contract.get("defer_reason_zh")
    if not isinstance(defer_reason, str):
        raise ValueError("risk_contract.defer_reason_zh must be a string")
    if defer_reason != defer_reason.strip():
        raise ValueError(
            "risk_contract.defer_reason_zh must not have surrounding whitespace"
        )
    if scope_status == "DEFERRED":
        if blocking:
            raise ValueError("deferred risk scope cannot be blocking")
        if not defer_reason:
            raise ValueError(
                "deferred risk scope requires a non-empty defer_reason_zh"
            )
    elif defer_reason:
        raise ValueError("active risk scope must not have defer_reason_zh")
    return (
        closure_id,
        rule_order,
        threshold,
        disclaimer,
        scope_status,
        blocking,
        defer_reason,
    )


def _facts_by_rule(facts_payload: object, allowed_rules: set[str]) -> dict[str, dict]:
    if not isinstance(facts_payload, dict):
        raise ValueError("facts root must be an object keyed by rule_id")
    unknown_rules = sorted(set(facts_payload) - allowed_rules)
    if unknown_rules:
        raise ValueError(f"unknown rules in facts: {unknown_rules}")

    normalized: dict[str, dict] = {}
    for rule_id, raw_entry in facts_payload.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"facts entry for {rule_id} must be an object")
        unknown_fields = sorted(set(raw_entry) - {"facts", "subject_ids"})
        if unknown_fields:
            raise ValueError(f"unknown fields for {rule_id}: {unknown_fields}")

        rule_facts = raw_entry.get("facts", {})
        if not isinstance(rule_facts, dict):
            raise ValueError(f"facts for {rule_id} must be an object")
        if any(not isinstance(key, str) for key in rule_facts):
            raise ValueError(f"fact keys for {rule_id} must be strings")

        subject_ids = raw_entry.get("subject_ids", [])
        if not isinstance(subject_ids, list) or any(
            not isinstance(subject_id, str) or not subject_id
            for subject_id in subject_ids
        ):
            raise ValueError(f"subject_ids for {rule_id} must be non-empty strings")
        normalized[rule_id] = {
            "facts": rule_facts,
            "subject_ids": subject_ids,
        }
    return normalized


def build_outputs(
    closure: object,
    facts_payload: object,
    *,
    closure_sha256: str,
    facts_sha256: str,
) -> tuple[bytes, bytes]:
    (
        closure_id,
        rule_order,
        threshold,
        disclaimer,
        scope_status,
        blocking,
        defer_reason,
    ) = _closure_contract(closure)
    by_rule = _facts_by_rule(facts_payload, set(rule_order))

    assessments = []
    for rule_id in rule_order:
        # 缺失规则仍调用同一个评估器，不在 CLI 内伪造结论。
        entry = by_rule.get(rule_id, {"facts": {}, "subject_ids": []})
        assessment = evaluate_risk_rule(
            rule_id,
            entry["facts"],
            subject_ids=entry["subject_ids"],
            min_evidence_confidence=threshold,
        )
        assessments.append(assessment.model_dump(mode="json"))

    assessments_payload = {
        "schema_version": "1.0",
        "closure_id": closure_id,
        "scope_status": scope_status,
        "blocking": blocking,
        "defer_reason_zh": defer_reason,
        "rule_order": rule_order,
        "assessments": assessments,
    }
    assessments_bytes = _canonical_json_bytes(assessments_payload)

    counts = {status.value: 0 for status in RiskStatus}
    for assessment in assessments:
        counts[assessment["status"]] += 1
    metrics_payload = {
        "schema_version": "1.0",
        "closure_id": closure_id,
        "scope_status": scope_status,
        "blocking": blocking,
        "defer_reason_zh": defer_reason,
        "rule_count": len(rule_order),
        "rule_order": rule_order,
        "status_counts": counts,
        "disclaimer_zh": disclaimer,
        "input_sha256": {
            "closure": closure_sha256,
            "facts": facts_sha256,
        },
        "output_sha256": {
            "assessments.json": _sha256_bytes(assessments_bytes),
        },
    }
    return assessments_bytes, _canonical_json_bytes(metrics_payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closure", required=True, type=Path)
    ap.add_argument("--facts", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    closure, closure_bytes = _read_json(args.closure)
    facts_payload, facts_bytes = _read_json(args.facts)
    assessments_bytes, metrics_bytes = build_outputs(
        closure,
        facts_payload,
        closure_sha256=_sha256_bytes(closure_bytes),
        facts_sha256=_sha256_bytes(facts_bytes),
    )

    # 所有解析、合同核对和规则评估成功后才创建目录，保持 fail-closed。
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "assessments.json").write_bytes(assessments_bytes)
    (args.out_dir / "metrics.json").write_bytes(metrics_bytes)
    print(
        json.dumps(
            {
                "rules": 3,
                "assessments_sha256": _sha256_bytes(assessments_bytes),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
