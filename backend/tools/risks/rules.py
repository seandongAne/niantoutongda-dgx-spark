"""三条英雄切片辅助风险提醒规则。

这些规则只消费已经提取并带画面引用的布尔事实，不从标签或位置名称猜测
安全结论。每条规则按冻结顺序检查事实：

* 事实缺失、没有证据引用或低于置信度门槛 -> ``NEEDS_USER``；
* 某个充分取证的事实为否 -> ``NOT_APPLICABLE``；
* 全部事实充分取证且为真 -> ``TRIGGERED``。

因此，``TRIGGERED`` 永远不会由部分证据产生。输出始终携带本次实际使用
的事实、证据引用、结论置信度及“非安全认证”声明。
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MIN_EVIDENCE_CONFIDENCE = 0.8
RISK_DISCLAIMER_ZH = (
    "仅为辅助风险提醒，不构成安全认证，也不能替代现场人员或专业人员复核。"
)

RULE_FACT_KEYS: dict[str, tuple[str, ...]] = {
    "CHILD_SHARP_TOOL_REACH": (
        "child_present",
        "sharp_tool_present",
        "within_child_reach",
    ),
    "TRIP_HAZARD_IN_PATH": (
        "trip_hazard_present",
        "in_walk_path",
    ),
    "POWER_IN_WET_ZONE": (
        "powered_item_present",
        "wet_zone_present",
        "in_wet_zone",
    ),
}


class RiskStatus(str, Enum):
    TRIGGERED = "TRIGGERED"
    NEEDS_USER = "NEEDS_USER"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RiskEvidence(BaseModel):
    """一个可审计的布尔事实；``None`` 表示尚无结论。"""

    model_config = ConfigDict(extra="forbid")

    value: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_refs: list[str] = []


class RiskAssessment(BaseModel):
    """规则输出；不提供安全认证或无风险保证。"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    status: RiskStatus
    subject_ids: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, RiskEvidence]
    evidence_refs: list[str]
    reason_codes: list[str]
    disclaimer_zh: str = RISK_DISCLAIMER_ZH


def _coerce_evidence(value: RiskEvidence | Mapping[str, object]) -> RiskEvidence:
    if isinstance(value, RiskEvidence):
        return value
    return RiskEvidence.model_validate(value)


def _decision_confidence(
    evidence: Mapping[str, RiskEvidence],
    evaluated_keys: list[str],
    *,
    incomplete: bool,
) -> float:
    if incomplete:
        # NEEDS_USER 是“无法自动下结论”，结论置信度固定为零；各事实原始
        # 置信度仍完整保留在 evidence 中供 UI 展示和人工复核。
        return 0.0
    confidences = [
        evidence[key].confidence
        for key in evaluated_keys
        if evidence[key].confidence is not None
    ]
    return min(confidences, default=0.0)


def evaluate_risk_rule(
    rule_id: str,
    facts: Mapping[str, RiskEvidence | Mapping[str, object]],
    *,
    subject_ids: list[str] | tuple[str, ...] = (),
    min_evidence_confidence: float = DEFAULT_MIN_EVIDENCE_CONFIDENCE,
) -> RiskAssessment:
    """以冻结的事实顺序评估一条辅助风险提醒。

    ``NOT_APPLICABLE`` 只表示当前规则的触发条件被充分的否定证据阻断，
    不是“场景安全”。未知规则、未知事实键和非法门槛均显式失败，避免配置
    漂移被静默忽略。
    """

    if rule_id not in RULE_FACT_KEYS:
        raise ValueError(f"unknown risk rule: {rule_id}")
    if not 0.0 <= min_evidence_confidence <= 1.0:
        raise ValueError("min_evidence_confidence must be within [0, 1]")

    required_keys = RULE_FACT_KEYS[rule_id]
    unknown_keys = sorted(set(facts) - set(required_keys))
    if unknown_keys:
        raise ValueError(f"unexpected facts for {rule_id}: {unknown_keys}")

    normalized = {
        key: _coerce_evidence(facts[key])
        for key in required_keys
        if key in facts
    }
    evaluated_keys: list[str] = []
    status = RiskStatus.TRIGGERED
    reason_codes: list[str] = []
    incomplete = False

    for key in required_keys:
        fact = normalized.get(key)
        if fact is None:
            status = RiskStatus.NEEDS_USER
            reason_codes = [f"MISSING_EVIDENCE:{key}"]
            incomplete = True
            break

        evaluated_keys.append(key)
        if fact.value is None:
            status = RiskStatus.NEEDS_USER
            reason_codes = [f"UNKNOWN_FACT:{key}"]
            incomplete = True
            break
        if not fact.evidence_refs:
            status = RiskStatus.NEEDS_USER
            reason_codes = [f"MISSING_EVIDENCE_REF:{key}"]
            incomplete = True
            break
        if fact.confidence is None:
            status = RiskStatus.NEEDS_USER
            reason_codes = [f"MISSING_CONFIDENCE:{key}"]
            incomplete = True
            break
        if fact.confidence < min_evidence_confidence:
            status = RiskStatus.NEEDS_USER
            reason_codes = [f"LOW_CONFIDENCE:{key}"]
            incomplete = True
            break
        if fact.value is False:
            status = RiskStatus.NOT_APPLICABLE
            reason_codes = [f"NEGATED_TRIGGER_FACT:{key}"]
            break

    if status == RiskStatus.TRIGGERED:
        reason_codes = [f"RULE_TRIGGERED:{rule_id}"]

    used_evidence = {
        key: normalized[key]
        for key in evaluated_keys
    }
    evidence_refs = sorted(
        {
            ref
            for fact in used_evidence.values()
            for ref in fact.evidence_refs
        }
    )
    return RiskAssessment(
        rule_id=rule_id,
        status=status,
        subject_ids=sorted(set(subject_ids)),
        confidence=_decision_confidence(
            used_evidence,
            evaluated_keys,
            incomplete=incomplete,
        ),
        evidence=used_evidence,
        evidence_refs=evidence_refs,
        reason_codes=reason_codes,
    )
