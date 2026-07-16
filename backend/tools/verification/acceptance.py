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
    VerificationCheckRequest,
    VerificationVerdict,
    compute_payload_hash,
)
from backend.schemas.hero_bundle import AcceptanceManifest, TaskCard
from backend.tools.verification.verdict import derive_verdict


@dataclass
class CardVerification:
    card: TaskCard
    request: VerificationCheckRequest
    presence: ObjectPresenceCheckResult
    compliance: PlacementComplianceResult
    verdict: VerificationVerdict


def _finalize(msg):
    msg.payload_hash = compute_payload_hash(msg)
    return msg


def verify_card(card: TaskCard, acceptance: AcceptanceManifest) -> CardVerification:
    corr = f"verify-{card.card_id}"
    expected = [item.entity_id for item in card.items]

    # 该卡相关照片 = 拍目标区域的照片 + 任何拍到了本卡实体的照片
    # (拍错区域的照片也是证据——它证明实体放错了地方)。
    relevant = [
        p
        for p in acceptance.photos
        if p.region_id == card.target_region_id
        or any(m.entity_id in expected and m.present for m in p.matches)
    ]

    request = _finalize(
        VerificationCheckRequest(
            message_id=f"{corr}-request",
            correlation_id=corr,
            producer=AgentRole.EXEC,
            task_id=card.card_id,
            expected_entity_ids=expected,
            target_region_id=card.target_region_id,
            photo_refs=[p.photo_ref for p in relevant],
        )
    )

    presences: list[EntityPresence] = []
    compliances: list[PlacementCompliance] = []
    for entity_id in expected:
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
            seen_regions = sorted({photo.region_id for photo, _ in hits})
            region_ok = card.target_region_id in seen_regions
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
            presences.append(EntityPresence(entity_id=entity_id, present=False))
            compliances.append(
                PlacementCompliance(
                    entity_id=entity_id,
                    region_ok=False,
                    relations_ok=True,
                    violated_constraints=["NOT_IN_ANY_PHOTO"],
                )
            )

    presence = _finalize(
        ObjectPresenceCheckResult(
            message_id=f"{corr}-presence",
            correlation_id=corr,
            causation_id=request.message_id,
            producer=AgentRole.MEM,
            request_id=request.message_id,
            presences=presences,
        )
    )
    compliance = _finalize(
        PlacementComplianceResult(
            message_id=f"{corr}-compliance",
            correlation_id=corr,
            causation_id=request.message_id,
            producer=AgentRole.SPACE,
            request_id=request.message_id,
            compliances=compliances,
        )
    )
    verdict = derive_verdict(
        request, presence, compliance, verdict_id=f"{corr}-verdict"
    )
    return CardVerification(card, request, presence, compliance, verdict)


def verify_cards(
    cards: list[TaskCard], acceptance: AcceptanceManifest
) -> list[CardVerification]:
    return [verify_card(card, acceptance) for card in cards]


def card_status_after(verification: CardVerification) -> TaskStatus:
    """只有 VERIFIED 改卡状态;FAILED/NEEDS_USER 留给用户裁决,不代写。"""
    if verification.verdict.verdict == "VERIFIED":
        return TaskStatus.VERIFIED
    return verification.card.status
