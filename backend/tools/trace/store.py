"""不可变 Agent 消息的 JSONL 存储、合并与协议回放。

每个流水线阶段先写自己的 trace fragment，最终按阶段顺序合并成
``audit/events.jsonl``。fragment 避免断点续跑时共享文件被重复追加；最终
events.jsonl 仍是逐行追加语义，消息一经写入不允许修改或重复 message_id。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from backend.schemas.core import (
    AgentHandoff,
    AgentRole,
    ClarificationDecision,
    ClarificationRequest,
    ObjectPresenceCheckResult,
    PlacementComplianceResult,
    UserAdjudication,
    VerificationCheckRequest,
    VerificationVerdict,
    _Message,
    compute_payload_hash,
)


class TraceValidationError(ValueError):
    """trace 被篡改、断链或不满足严格完成门。"""


MESSAGE_TYPES: dict[str, type[_Message]] = {
    cls.__name__: cls
    for cls in (
        AgentHandoff,
        ClarificationRequest,
        ClarificationDecision,
        VerificationCheckRequest,
        ObjectPresenceCheckResult,
        PlacementComplianceResult,
        VerificationVerdict,
        UserAdjudication,
    )
}

# 兼容早期 verify/messages.jsonl 的短 type；新产物统一写 message_type。
LEGACY_TYPES = {
    "request": "VerificationCheckRequest",
    "presence": "ObjectPresenceCheckResult",
    "compliance": "PlacementComplianceResult",
    "verdict": "VerificationVerdict",
    "adjudication": "UserAdjudication",
}

MAIN_ACTIONS = (
    "ENTITIES_READY",
    "GROUPS_READY",
    "PLACEMENT_READY",
    "TASKS_READY",
)


def finalize_message(message: _Message) -> _Message:
    """就地写入 canonical payload hash 并返回消息，便于构造器串接。"""

    message.payload_hash = compute_payload_hash(message)
    return message


def _record(message: _Message) -> dict:
    return {
        "message_type": type(message).__name__,
        **message.model_dump(mode="json"),
    }


def write_fragment(path: str | Path, messages: Iterable[_Message]) -> None:
    """原子语义写一个阶段 fragment；每条消息必须已 finalize。"""

    rows = list(messages)
    for message in rows:
        if not message.payload_hash:
            raise TraceValidationError(f"{message.message_id}: empty payload_hash")
        if message.payload_hash != compute_payload_hash(message):
            raise TraceValidationError(f"{message.message_id}: payload_hash mismatch")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(_record(message), ensure_ascii=False, sort_keys=True) + "\n"
            for message in rows
        ),
        encoding="utf-8",
    )


def load_trace(path: str | Path) -> list[_Message]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    messages: list[_Message] = []
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"{target}:{line_no}: invalid JSON: {exc}") from exc
        name = row.pop("message_type", None)
        if name is None:
            name = LEGACY_TYPES.get(str(row.pop("type", "")))
        model = MESSAGE_TYPES.get(str(name))
        if model is None:
            raise TraceValidationError(f"{target}:{line_no}: unknown message_type {name!r}")
        try:
            messages.append(model.model_validate(row))
        except Exception as exc:
            raise TraceValidationError(f"{target}:{line_no}: {exc}") from exc
    return messages


def merge_fragments(paths: Iterable[str | Path], out: str | Path) -> list[_Message]:
    messages = [message for path in paths for message in load_trace(path)]
    # write_fragment 会再次验证 hash，避免把坏 fragment 拼进统一证据。
    write_fragment(out, messages)
    return messages


def require_handoff(path: str | Path, action: str) -> AgentHandoff:
    matches = [
        message
        for message in load_trace(path)
        if isinstance(message, AgentHandoff) and message.action == action
    ]
    if len(matches) != 1:
        raise TraceValidationError(
            f"{path}: expected exactly one {action} handoff, got {len(matches)}"
        )
    return matches[0]


def _one(items: list[_Message], label: str, request_id: str) -> _Message:
    if len(items) != 1:
        raise TraceValidationError(
            f"{request_id}: expected exactly one {label}, got {len(items)}"
        )
    return items[0]


def validate_trace(
    messages: Iterable[_Message],
    *,
    require_main_chain: bool = False,
    require_verification: bool = False,
    require_closed_choices: bool = False,
    require_adjudication: bool = False,
) -> dict:
    """回放并验证 hash、因果 DAG、主链路由和验收协议闭合。"""

    rows = list(messages)
    by_id: dict[str, _Message] = {}
    ordered_ids: set[str] = set()
    for index, message in enumerate(rows, start=1):
        if not message.message_id:
            raise TraceValidationError(f"line {index}: empty message_id")
        if message.message_id in by_id:
            raise TraceValidationError(f"duplicate message_id: {message.message_id}")
        if not message.payload_hash:
            raise TraceValidationError(f"{message.message_id}: empty payload_hash")
        if message.payload_hash != compute_payload_hash(message):
            raise TraceValidationError(f"{message.message_id}: payload_hash mismatch")
        if message.causation_id:
            if message.causation_id not in ordered_ids:
                raise TraceValidationError(
                    f"{message.message_id}: causation_id {message.causation_id} is missing or not earlier"
                )
            parent = by_id[message.causation_id]
            cross_correlation_root = (
                isinstance(message, VerificationCheckRequest)
                and isinstance(parent, AgentHandoff)
                and parent.action == "TASKS_READY"
            )
            if not cross_correlation_root and parent.correlation_id != message.correlation_id:
                raise TraceValidationError(
                    f"{message.message_id}: causation crosses correlation without child-root permission"
                )
        by_id[message.message_id] = message
        ordered_ids.add(message.message_id)

    # 四 Agent 主链必须是同一 correlation 下的固定交接序列。
    handoffs_by_corr: dict[str, list[AgentHandoff]] = defaultdict(list)
    for message in rows:
        if isinstance(message, AgentHandoff):
            handoffs_by_corr[message.correlation_id].append(message)
    complete_main = []
    for correlation_id, handoffs in handoffs_by_corr.items():
        actions = tuple(message.action for message in handoffs)
        if actions == MAIN_ACTIONS:
            for previous, current in zip(handoffs, handoffs[1:]):
                if current.causation_id != previous.message_id:
                    raise TraceValidationError(
                        f"{correlation_id}: {current.action} does not causally follow {previous.action}"
                    )
            complete_main.append(correlation_id)
    if require_main_chain and len(complete_main) != 1:
        raise TraceValidationError(
            f"expected exactly one complete four-Agent chain, got {len(complete_main)}"
        )

    clarification_requests = [m for m in rows if isinstance(m, ClarificationRequest)]
    clarification_decisions = [m for m in rows if isinstance(m, ClarificationDecision)]
    decisions_by_request: dict[str, list[ClarificationDecision]] = defaultdict(list)
    for decision in clarification_decisions:
        decisions_by_request[decision.request_id].append(decision)
        request = by_id.get(decision.request_id)
        if not isinstance(request, ClarificationRequest):
            raise TraceValidationError(
                f"{decision.message_id}: request_id does not reference ClarificationRequest"
            )
        if decision.causation_id != request.message_id:
            raise TraceValidationError(f"{decision.message_id}: decision causation mismatch")
    closed_choices = 0
    for request in clarification_requests:
        decisions = decisions_by_request.get(request.message_id, [])
        if len(decisions) > 1:
            raise TraceValidationError(f"{request.message_id}: multiple clarification decisions")
        closed_choices += int(len(decisions) == 1)
    open_choices = len(clarification_requests) - closed_choices
    if require_closed_choices and (not clarification_requests or open_choices):
        raise TraceValidationError(
            f"clarification choices not closed: requests={len(clarification_requests)}, open={open_choices}"
        )

    verify_requests = [m for m in rows if isinstance(m, VerificationCheckRequest)]
    presence_by_request: dict[str, list[ObjectPresenceCheckResult]] = defaultdict(list)
    compliance_by_request: dict[str, list[PlacementComplianceResult]] = defaultdict(list)
    verdict_by_request: dict[str, list[VerificationVerdict]] = defaultdict(list)
    adjudication_by_verdict: dict[str, list[UserAdjudication]] = defaultdict(list)
    for message in rows:
        if isinstance(message, ObjectPresenceCheckResult):
            presence_by_request[message.request_id].append(message)
        elif isinstance(message, PlacementComplianceResult):
            compliance_by_request[message.request_id].append(message)
        elif isinstance(message, VerificationVerdict):
            verdict_by_request[message.request_id].append(message)
        elif isinstance(message, UserAdjudication):
            adjudication_by_verdict[message.verdict_id].append(message)

    closed_verifications = 0
    required_adjudications = 0
    closed_adjudications = 0
    for request in verify_requests:
        presence = _one(
            presence_by_request.get(request.message_id, []), "presence result", request.message_id
        )
        compliance = _one(
            compliance_by_request.get(request.message_id, []), "compliance result", request.message_id
        )
        verdict = _one(
            verdict_by_request.get(request.message_id, []), "verdict", request.message_id
        )
        assert isinstance(presence, ObjectPresenceCheckResult)
        assert isinstance(compliance, PlacementComplianceResult)
        assert isinstance(verdict, VerificationVerdict)
        if verdict.presence_result_id != presence.message_id:
            raise TraceValidationError(f"{verdict.message_id}: presence_result_id mismatch")
        if verdict.compliance_result_id != compliance.message_id:
            raise TraceValidationError(f"{verdict.message_id}: compliance_result_id mismatch")
        if {presence.correlation_id, compliance.correlation_id, verdict.correlation_id} != {
            request.correlation_id
        }:
            raise TraceValidationError(f"{request.message_id}: verification correlation not closed")
        closed_verifications += 1
        if verdict.verdict != "VERIFIED":
            required_adjudications += 1
            adjudications = adjudication_by_verdict.get(verdict.message_id, [])
            if len(adjudications) > 1:
                raise TraceValidationError(f"{verdict.message_id}: multiple user adjudications")
            if adjudications:
                adjudication = adjudications[0]
                if adjudication.causation_id != verdict.message_id:
                    raise TraceValidationError(
                        f"{adjudication.message_id}: adjudication causation mismatch"
                    )
                closed_adjudications += 1
    if require_verification and not verify_requests:
        raise TraceValidationError("no verification chain found")
    if require_adjudication and closed_adjudications != required_adjudications:
        raise TraceValidationError(
            "non-VERIFIED verdicts are not fully adjudicated: "
            f"required={required_adjudications}, closed={closed_adjudications}"
        )

    return {
        "status": "PASS",
        "message_count": len(rows),
        "message_types": dict(sorted(Counter(type(m).__name__ for m in rows).items())),
        "producer_counts": dict(sorted(Counter(m.producer.value for m in rows).items())),
        "correlation_count": len({m.correlation_id for m in rows}),
        "main_chain": {
            "complete": len(complete_main),
            "correlation_ids": complete_main,
            "actions": list(MAIN_ACTIONS),
        },
        "clarifications": {
            "requests": len(clarification_requests),
            "closed": closed_choices,
            "open": open_choices,
        },
        "verification": {
            "requests": len(verify_requests),
            "closed": closed_verifications,
            "adjudication_required": required_adjudications,
            "adjudication_closed": closed_adjudications,
        },
    }
