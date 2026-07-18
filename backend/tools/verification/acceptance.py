"""验收照片 → 验收复核消息族 — G7b 切片的确定性执行器。

职责严格三分(verdict.py 的协议在此复用,不另立裁决口径):
- MEM 适配:把照片匹配翻译成 ObjectPresenceCheckResult,只答"出现与否";
  照片没覆盖到的实体一律 present=false,绝不因缺照片静默通过。
- SPACE 确定性校验:只答"该实体出现的照片是否就是目标区域的照片";
  出现在别的区域记 WRONG_REGION,一张照片都没出现记 NOT_IN_ANY_PHOTO。
- EXEC 汇总走 derive_verdict,VERIFIED 必须 presence ∧ compliance。

消息 id 全部由 card_id 确定性派生,同输入两次运行产生逐字节相同的
消息链(payload_hash 一致),满足复跑指纹要求。
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.schemas.core import (
    AgentRole,
    ObjectPresenceCheckResult,
    EntityPresence,
    PlacementCompliance,
    PlacementComplianceResult,
    TaskStatus,
    UserAdjudication,
    VerificationCheckRequest,
    VerificationVerdict,
    compute_payload_hash,
)
from backend.schemas.hero_bundle import AcceptanceManifest, AcceptancePhoto, TaskCard
from backend.tools.verification.verdict import derive_verdict


@dataclass
class CardVerification:
    card: TaskCard
    request: VerificationCheckRequest
    presence: ObjectPresenceCheckResult
    compliance: PlacementComplianceResult
    verdict: VerificationVerdict
    adjudication: UserAdjudication | None = None


def _finalize(msg):
    msg.payload_hash = compute_payload_hash(msg)
    return msg


def _require_valid_payload_hash(msg) -> None:
    if not msg.payload_hash:
        raise ValueError(f"{msg.message_id}: empty payload_hash")
    if msg.payload_hash != compute_payload_hash(msg):
        raise ValueError(f"{msg.message_id}: payload_hash mismatch")


def _validate_request_envelope(request: VerificationCheckRequest) -> None:
    _require_valid_payload_hash(request)
    if request.producer != AgentRole.EXEC:
        raise ValueError(f"{request.message_id}: request producer must be EXEC")
    if len(request.expected_entity_ids) != len(set(request.expected_entity_ids)):
        raise ValueError(f"{request.message_id}: duplicate expected_entity_id")


def _expected_entity_ids(card: TaskCard) -> list[str]:
    expected = [item.entity_id for item in card.items]
    if len(expected) != len(set(expected)):
        raise ValueError(f"{card.card_id}: duplicate entity_id in task card")
    return expected


def _photos_for_request(
    request: VerificationCheckRequest,
    acceptance: AcceptanceManifest,
) -> list[AcceptancePhoto]:
    if len(request.photo_refs) != len(set(request.photo_refs)):
        raise ValueError(f"{request.message_id}: duplicate photo_ref in request")
    by_ref = {photo.photo_ref: photo for photo in acceptance.photos}
    missing = [ref for ref in request.photo_refs if ref not in by_ref]
    if missing:
        raise ValueError(
            f"{request.message_id}: acceptance manifest missing photos: {missing}"
        )
    return [by_ref[ref] for ref in request.photo_refs]


def build_verification_request(
    card: TaskCard,
    acceptance: AcceptanceManifest,
    *,
    parent_message_id: str | None = None,
) -> VerificationCheckRequest:
    """EXEC 构造双路复核请求，不生成或预判任一路复核结果。"""

    if not acceptance.includes_card(card.card_id):
        raise ValueError(f"{card.card_id}: task card is outside acceptance scope")
    corr = f"verify-{card.card_id}"
    expected = _expected_entity_ids(card)

    # 该卡相关照片 = 拍目标区域的照片 + 任何拍到了本卡实体的照片
    # (拍错区域的照片也是证据——它证明实体放错了地方)。
    relevant = [
        p
        for p in acceptance.photos
        if p.region_id == card.target_region_id
        or any(m.entity_id in expected and m.present for m in p.matches)
    ]

    return _finalize(
        VerificationCheckRequest(
            message_id=f"{corr}-request",
            correlation_id=corr,
            causation_id=parent_message_id,
            producer=AgentRole.EXEC,
            task_id=card.card_id,
            expected_entity_ids=expected,
            target_region_id=card.target_region_id,
            photo_refs=[p.photo_ref for p in relevant],
        )
    )


def build_presence_result(
    request: VerificationCheckRequest,
    acceptance: AcceptanceManifest,
) -> ObjectPresenceCheckResult:
    """MEM 独立回答 request 中每个实体是否在请求照片内出现。"""

    _validate_request_envelope(request)
    relevant = _photos_for_request(request, acceptance)
    presences: list[EntityPresence] = []
    for entity_id in request.expected_entity_ids:
        hits = [
            (photo, match)
            for photo in relevant
            for match in photo.matches
            if match.entity_id == entity_id and match.present
        ]
        if hits:
            best_photo, best = max(
                hits, key=lambda pm: (pm[1].match_score is not None, pm[1].match_score or 0.0)
            )
            presences.append(
                EntityPresence(
                    entity_id=entity_id,
                    present=True,
                    match_score=best.match_score,
                    evidence_refs=[best_photo.photo_ref, *best.evidence_refs],
                )
            )
        else:
            presences.append(EntityPresence(entity_id=entity_id, present=False))

    return _finalize(
        ObjectPresenceCheckResult(
            message_id=f"{request.correlation_id}-presence",
            correlation_id=request.correlation_id,
            causation_id=request.message_id,
            producer=AgentRole.MEM,
            request_id=request.message_id,
            presences=presences,
        )
    )


def build_compliance_result(
    request: VerificationCheckRequest,
    acceptance: AcceptanceManifest,
) -> PlacementComplianceResult:
    """SPACE 独立回答 request 中每个实体的目标区域约束是否满足。"""

    _validate_request_envelope(request)
    relevant = _photos_for_request(request, acceptance)
    compliances: list[PlacementCompliance] = []
    for entity_id in request.expected_entity_ids:
        hits = [
            (photo, match)
            for photo in relevant
            for match in photo.matches
            if match.entity_id == entity_id and match.present
        ]
        if hits:
            seen_regions = sorted({photo.region_id for photo, _ in hits})
            region_ok = request.target_region_id in seen_regions
            compliances.append(
                PlacementCompliance(
                    entity_id=entity_id,
                    region_ok=region_ok,
                    relations_ok=True,
                    violated_constraints=(
                        [] if region_ok else [f"WRONG_REGION:{'/'.join(seen_regions)}"]
                    ),
                )
            )
        else:
            compliances.append(
                PlacementCompliance(
                    entity_id=entity_id,
                    region_ok=False,
                    relations_ok=True,
                    violated_constraints=["NOT_IN_ANY_PHOTO"],
                )
            )

    return _finalize(
        PlacementComplianceResult(
            message_id=f"{request.correlation_id}-compliance",
            correlation_id=request.correlation_id,
            causation_id=request.message_id,
            producer=AgentRole.SPACE,
            request_id=request.message_id,
            compliances=compliances,
        )
    )


def validate_verification_request(
    card: TaskCard,
    request: VerificationCheckRequest,
) -> None:
    """在 worker/fan-in 边界校验 EXEC request 未偏离冻结任务卡。"""

    _validate_request_envelope(request)
    expected = _expected_entity_ids(card)
    if request.task_id != card.card_id:
        raise ValueError(f"{request.message_id}: task_id does not match task card")
    if request.expected_entity_ids != expected:
        raise ValueError(
            f"{request.message_id}: expected_entity_ids do not match task card"
        )
    if request.target_region_id != card.target_region_id:
        raise ValueError(f"{request.message_id}: target_region_id does not match task card")


def finalize_verification(
    card: TaskCard,
    request: VerificationCheckRequest,
    presence: ObjectPresenceCheckResult,
    compliance: PlacementComplianceResult,
    acceptance: AcceptanceManifest,
) -> CardVerification:
    """EXEC fan-in：两路正式结果齐备后唯一生成 verdict/adjudication。"""

    if not acceptance.includes_card(card.card_id):
        raise ValueError(f"{card.card_id}: task card is outside acceptance scope")
    validate_verification_request(card, request)
    for label, result, producer in (
        ("presence", presence, AgentRole.MEM),
        ("compliance", compliance, AgentRole.SPACE),
    ):
        _require_valid_payload_hash(result)
        if result.producer != producer:
            raise ValueError(
                f"{result.message_id}: {label} producer must be {producer.value}"
            )
        if result.causation_id != request.message_id:
            raise ValueError(
                f"{result.message_id}: {label} causation_id must reference request"
            )
    verdict = derive_verdict(
        request,
        presence,
        compliance,
        verdict_id=f"{request.correlation_id}-verdict",
    )
    decision = next(
        (item for item in acceptance.adjudications if item.card_id == card.card_id),
        None,
    )
    if decision and verdict.verdict == "VERIFIED":
        raise ValueError(f"{card.card_id}: VERIFIED verdict cannot be user-overridden")
    adjudication = None
    if decision:
        adjudication = _finalize(
            UserAdjudication(
                message_id=f"{request.correlation_id}-adjudication",
                correlation_id=request.correlation_id,
                causation_id=verdict.message_id,
                producer=AgentRole.USER,
                verdict_id=verdict.message_id,
                decision=decision.decision,
                note=decision.note,
            )
        )
    return CardVerification(
        card, request, presence, compliance, verdict, adjudication
    )


def verify_card(
    card: TaskCard,
    acceptance: AcceptanceManifest,
    *,
    parent_message_id: str | None = None,
) -> CardVerification:
    """兼容入口：按既有确定性顺序组合 EXEC、MEM、SPACE 与 verdict。"""

    request = build_verification_request(
        card,
        acceptance,
        parent_message_id=parent_message_id,
    )
    presence = build_presence_result(request, acceptance)
    compliance = build_compliance_result(request, acceptance)
    return finalize_verification(card, request, presence, compliance, acceptance)


def verify_cards(
    cards: list[TaskCard],
    acceptance: AcceptanceManifest,
    *,
    parent_message_id: str | None = None,
) -> list[CardVerification]:
    card_ids = [card.card_id for card in cards]
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("duplicate task card_id")
    if acceptance.selected_card_ids is not None:
        missing = sorted(set(acceptance.selected_card_ids) - set(card_ids))
        if missing:
            raise ValueError(f"selected task cards missing from input: {missing}")
    return [
        verify_card(card, acceptance, parent_message_id=parent_message_id)
        for card in cards
        if acceptance.includes_card(card.card_id)
    ]


def card_status_after(verification: CardVerification) -> TaskStatus:
    """只有 VERIFIED 改卡状态;FAILED/NEEDS_USER 留给用户裁决,不代写。"""
    if verification.verdict.verdict == "VERIFIED":
        return TaskStatus.VERIFIED
    if (
        verification.adjudication
        and verification.adjudication.decision == "accept_override"
    ):
        return TaskStatus.USER_OVERRIDDEN
    return verification.card.status
