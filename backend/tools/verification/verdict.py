"""EXEC 侧验收裁决 — VERIFIED 的唯一入口。

规则(设计文档 §6.8):
- MEM 的 ObjectPresenceCheckResult 只证明"物品出现";
- SPACE 的 PlacementComplianceResult 只证明"区域与关系满足";
- 两者对同一 VerificationCheckRequest 都通过,EXEC 才能给 VERIFIED。
任何一方缺席、答非所问(request_id/correlation 不符)或未覆盖全部
expected_entity_ids,都是协议错误直接抛异常,不得静默降级成结论。
"""

from __future__ import annotations

from backend.schemas.core import (
    AgentRole,
    ObjectPresenceCheckResult,
    PlacementComplianceResult,
    VerificationCheckRequest,
    VerificationVerdict,
    compute_payload_hash,
)

LOW_CONFIDENCE_THRESHOLD = 0.6


def derive_verdict(
    request: VerificationCheckRequest,
    presence: ObjectPresenceCheckResult,
    compliance: PlacementComplianceResult,
    *,
    verdict_id: str,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> VerificationVerdict:
    for result in (presence, compliance):
        if result.request_id != request.message_id:
            raise ValueError(
                f"request_id mismatch: {result.request_id} != {request.message_id}"
            )
        if result.correlation_id != request.correlation_id:
            raise ValueError(
                f"correlation_id mismatch: {result.correlation_id} != {request.correlation_id}"
            )

    expected = set(request.expected_entity_ids)
    presence_by_id = {p.entity_id: p for p in presence.presences}
    compliance_by_id = {c.entity_id: c for c in compliance.compliances}
    if missing_cov := expected - set(presence_by_id):
        raise ValueError(f"presence result missing entities: {sorted(missing_cov)}")
    if missing_cov := expected - set(compliance_by_id):
        raise ValueError(f"compliance result missing entities: {sorted(missing_cov)}")

    reason_codes: list[str] = []
    needs_user: list[str] = []
    for entity_id in sorted(expected):
        p = presence_by_id[entity_id]
        c = compliance_by_id[entity_id]
        if not p.present:
            reason_codes.append(f"NOT_SEEN:{entity_id}")
            continue
        if not (c.region_ok and c.relations_ok):
            detail = ",".join(c.violated_constraints) or "unspecified"
            reason_codes.append(f"MISPLACED:{entity_id}:{detail}")
            continue
        if p.match_score is not None and p.match_score < low_confidence_threshold:
            needs_user.append(f"LOW_CONFIDENCE:{entity_id}")

    if reason_codes:
        verdict = "FAILED"
        reason_codes.extend(needs_user)
    elif needs_user:
        verdict = "NEEDS_USER"
        reason_codes = needs_user
    else:
        verdict = "VERIFIED"

    out = VerificationVerdict(
        message_id=verdict_id,
        correlation_id=request.correlation_id,
        # 约定:verdict 在双结果齐备时发出,causation 指向 compliance(后置校验)
        causation_id=compliance.message_id,
        producer=AgentRole.EXEC,
        request_id=request.message_id,
        presence_result_id=presence.message_id,
        compliance_result_id=compliance.message_id,
        verdict=verdict,
        reason_codes=reason_codes,
    )
    out.payload_hash = compute_payload_hash(out)
    return out
