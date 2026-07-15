"""验收复核消息族 + EXEC 裁决门的契约测试。

覆盖评审 P0-1(出现≠摆对)与 P0-2(请求/结果/裁决必须是独立不可变消息)。
"""

import pytest
from pydantic import ValidationError

from backend.schemas.core import (
    AgentRole,
    EntityPresence,
    ObjectPresenceCheckResult,
    PlacementCompliance,
    PlacementComplianceResult,
    UserAdjudication,
    VerificationCheckRequest,
    compute_payload_hash,
)
from backend.tools.verification.verdict import derive_verdict


def _request(entities=("e1", "e2")) -> VerificationCheckRequest:
    return VerificationCheckRequest(
        message_id="req1",
        correlation_id="corr1",
        producer=AgentRole.EXEC,
        task_id="task1",
        expected_entity_ids=list(entities),
        target_region_id="r_desk",
        photo_refs=["p.jpg"],
    )


def _presence(present=True, score=0.9, request_id="req1", corr="corr1"):
    return ObjectPresenceCheckResult(
        message_id="pres1",
        correlation_id=corr,
        causation_id="req1",
        producer=AgentRole.MEM,
        request_id=request_id,
        presences=[
            EntityPresence(entity_id="e1", present=present, match_score=score),
            EntityPresence(entity_id="e2", present=True, match_score=0.95),
        ],
    )


def _compliance(region_ok=True, relations_ok=True, request_id="req1", corr="corr1"):
    return PlacementComplianceResult(
        message_id="comp1",
        correlation_id=corr,
        causation_id="req1",
        producer=AgentRole.SPACE,
        request_id=request_id,
        compliances=[
            PlacementCompliance(
                entity_id="e1",
                region_ok=region_ok,
                relations_ok=relations_ok,
                violated_constraints=[] if region_ok else ["wrong_region:r_bed"],
            ),
            PlacementCompliance(entity_id="e2", region_ok=True, relations_ok=True),
        ],
    )


def test_all_pass_gives_verified():
    v = derive_verdict(_request(), _presence(), _compliance(), verdict_id="v1")
    assert v.verdict == "VERIFIED"
    assert v.reason_codes == []
    assert v.correlation_id == "corr1"
    assert v.payload_hash == compute_payload_hash(v)


def test_present_but_misplaced_is_not_verified():
    # P0-1 核心场景:物品出现在照片里,但放错了区域 → 不得 VERIFIED
    v = derive_verdict(
        _request(), _presence(), _compliance(region_ok=False), verdict_id="v1"
    )
    assert v.verdict == "FAILED"
    assert any(code.startswith("MISPLACED:e1") for code in v.reason_codes)


def test_missing_entity_fails_with_not_seen():
    v = derive_verdict(
        _request(), _presence(present=False), _compliance(), verdict_id="v1"
    )
    assert v.verdict == "FAILED"
    assert "NOT_SEEN:e1" in v.reason_codes


def test_low_confidence_presence_needs_user():
    v = derive_verdict(
        _request(), _presence(score=0.4), _compliance(), verdict_id="v1"
    )
    assert v.verdict == "NEEDS_USER"
    assert "LOW_CONFIDENCE:e1" in v.reason_codes


def test_correlation_mismatch_is_protocol_error():
    with pytest.raises(ValueError, match="correlation_id"):
        derive_verdict(
            _request(), _presence(corr="corr_other"), _compliance(), verdict_id="v1"
        )
    with pytest.raises(ValueError, match="request_id"):
        derive_verdict(
            _request(), _presence(request_id="req_other"), _compliance(), verdict_id="v1"
        )


def test_partial_coverage_is_protocol_error():
    # 结果只答了部分物品 → 协议错误,不许静默成结论
    with pytest.raises(ValueError, match="missing entities"):
        derive_verdict(
            _request(entities=("e1", "e2", "e3")),
            _presence(),
            _compliance(),
            verdict_id="v1",
        )


def test_result_has_no_writable_conclusion_field():
    # P0-2:请求消息不再携带 result 回写位
    with pytest.raises(ValidationError):
        VerificationCheckRequest(
            message_id="req1",
            correlation_id="corr1",
            producer=AgentRole.EXEC,
            task_id="task1",
            expected_entity_ids=["e1"],
            target_region_id="r1",
            photo_refs=["p.jpg"],
            result="VERIFIED",
        )


def test_payload_hash_detects_tamper():
    v = derive_verdict(_request(), _presence(), _compliance(), verdict_id="v1")
    v.verdict = "FAILED"
    assert v.payload_hash != compute_payload_hash(v)


def test_user_adjudication_is_standalone_message():
    adj = UserAdjudication(
        message_id="adj1",
        correlation_id="corr1",
        causation_id="v1",
        producer=AgentRole.USER,
        verdict_id="v1",
        decision="accept_override",
        note="台灯放床头也行",
    )
    adj.payload_hash = compute_payload_hash(adj)
    assert UserAdjudication.model_validate_json(adj.model_dump_json()) == adj
