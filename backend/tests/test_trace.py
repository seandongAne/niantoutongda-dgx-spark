"""四 Agent trace：主链、二选一、验收与篡改检测。"""

from __future__ import annotations

import json

import pytest

from backend.schemas.core import (
    AgentHandoff,
    AgentRole,
    ClarificationDecision,
    ClarificationRequest,
)
from backend.schemas.hero_bundle import (
    AcceptanceAdjudication,
    AcceptanceManifest,
    AcceptanceMatch,
    AcceptancePhoto,
    TaskCard,
    TaskCardItem,
)
from backend.tools.trace import (
    TraceValidationError,
    finalize_message,
    load_trace,
    validate_trace,
    write_fragment,
)
from backend.tools.verification.acceptance import verify_card


def _handoff(message_id, correlation_id, cause, producer, target, action):
    return finalize_message(
        AgentHandoff(
            message_id=message_id,
            correlation_id=correlation_id,
            causation_id=cause,
            producer=producer,
            target=target,
            action=action,
            item_ids=[message_id],
        )
    )


def _complete_trace():
    corr = "hero-test"
    mem = _handoff(
        "mem", corr, None, AgentRole.MEM, AgentRole.GROUP, "ENTITIES_READY"
    )
    group = _handoff(
        "group", corr, "mem", AgentRole.GROUP, AgentRole.SPACE, "GROUPS_READY"
    )
    space = _handoff(
        "space", corr, "group", AgentRole.SPACE, AgentRole.EXEC, "PLACEMENT_READY"
    )
    execute = _handoff(
        "exec", corr, "space", AgentRole.EXEC, AgentRole.USER, "TASKS_READY"
    )
    request = finalize_message(
        ClarificationRequest(
            message_id="clarify",
            correlation_id="clarify-test",
            producer=AgentRole.MEM,
            request_id="clarify",
            candidate_a="t1",
            candidate_b="t2",
            reason_codes=["LOW_MARGIN"],
        )
    )
    decision = finalize_message(
        ClarificationDecision(
            message_id="clarify-decision",
            correlation_id="clarify-test",
            causation_id="clarify",
            producer=AgentRole.USER,
            request_id="clarify",
            decision="different",
        )
    )
    card = TaskCard(
        card_id="card-01",
        group_id="g1",
        box_label_zh="测试箱",
        items=[TaskCardItem(entity_id="e1", display_name_zh="台灯")],
        target_region_id="desk",
        target_region_name_zh="书桌",
    )
    acceptance = AcceptanceManifest(
        photos=[
            AcceptancePhoto(
                photo_ref="p.jpg",
                region_id="desk",
                matches=[
                    AcceptanceMatch(entity_id="e1", present=True, match_score=0.5)
                ],
            )
        ],
        adjudications=[
            AcceptanceAdjudication(
                card_id="card-01", decision="accept_override", note="确认"
            )
        ],
    )
    verified = verify_card(card, acceptance, parent_message_id=execute.message_id)
    return [
        request,
        decision,
        mem,
        group,
        space,
        execute,
        verified.request,
        verified.presence,
        verified.compliance,
        verified.verdict,
        verified.adjudication,
    ]


def test_complete_trace_strictly_replays(tmp_path):
    path = tmp_path / "events.jsonl"
    write_fragment(path, _complete_trace())
    report = validate_trace(
        load_trace(path),
        require_main_chain=True,
        require_verification=True,
        require_closed_choices=True,
        require_adjudication=True,
    )
    assert report["main_chain"]["complete"] == 1
    assert report["verification"]["closed"] == 1
    assert report["clarifications"]["closed"] == 1


def test_payload_tamper_is_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    write_fragment(path, _complete_trace())
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["candidate_a"] = "tampered"
    rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="payload_hash mismatch"):
        validate_trace(load_trace(path))


def test_missing_or_forward_causation_is_rejected():
    messages = _complete_trace()
    # group 现在位于其 parent(mem)之前。
    broken = [messages[0], messages[1], messages[3], messages[2], *messages[4:]]
    with pytest.raises(TraceValidationError, match="missing or not earlier"):
        validate_trace(broken)
