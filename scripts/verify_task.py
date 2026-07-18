#!/usr/bin/env python
"""验收复核阶段：EXEC fan-out → MEM/SPACE 独立 worker → EXEC fan-in。

FAILED/NEEDS_USER 是合法业务结局，不是管线错误（退出码仍为 0）；协议错误、
照片缺失、worker 超时/失败或任一路结果不完整都会 fail-closed，且不会生成最终
messages/verdicts。AcceptanceManifest.selected_card_ids 可把一次验收显式限制到
部分任务卡；未选择的任务卡保持原状态，不会被误判为失败。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import (  # noqa: E402
    AgentRole,
    ObjectPresenceCheckResult,
    PlacementComplianceResult,
    VerificationCheckRequest,
)
from backend.schemas.hero_bundle import AcceptanceManifest, TaskCard  # noqa: E402
from backend.tools.trace import load_trace, require_handoff, write_fragment  # noqa: E402
from backend.tools.verification.acceptance import (  # noqa: E402
    CardVerification,
    build_verification_request,
    card_status_after,
    finalize_verification,
)

ROLE_ORDER = ("MEM", "SPACE")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load_cards(path: Path) -> list[TaskCard]:
    cards = [
        TaskCard.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [card.card_id for card in cards]
    if not cards:
        raise ValueError(f"{path}: no task cards")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate task card_id")
    return cards


def _select_cards(
    cards: list[TaskCard], acceptance: AcceptanceManifest
) -> list[TaskCard]:
    by_id = {card.card_id: card for card in cards}
    if acceptance.selected_card_ids is not None:
        unknown = sorted(set(acceptance.selected_card_ids) - set(by_id))
        if unknown:
            raise ValueError(f"selected_card_ids not found in task cards: {unknown}")
    selected = [card for card in cards if acceptance.includes_card(card.card_id)]
    if not selected:
        raise ValueError("verification scope selected no task cards")
    return selected


def _resolve_photo_ref(photo_ref: str, photo_root: Path) -> Path:
    path = Path(photo_ref)
    resolved = path if path.is_absolute() else photo_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"verification photo missing: {photo_ref} ({resolved})")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"verification photo is empty: {photo_ref} ({resolved})")
    return resolved


def _validate_request_photos(
    requests: list[VerificationCheckRequest], photo_root: Path
) -> None:
    for request in requests:
        if not request.photo_refs:
            raise ValueError(
                f"{request.message_id}: no relevant verification photos; refusing verdict"
            )
        for photo_ref in request.photo_refs:
            _resolve_photo_ref(photo_ref, photo_root)


def _write_telemetry(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_parallel_workers(
    commands: dict[str, list[str]],
    *,
    timeout_seconds: float,
    telemetry_path: Path,
    cwd: Path = PROJ,
) -> dict[str, dict]:
    """先启动全部 role，再轮询收敛；任一路失败/超时即整体失败。"""

    if tuple(commands) != ROLE_ORDER:
        raise ValueError(f"worker roles must be ordered as {ROLE_ORDER}")
    if timeout_seconds <= 0:
        raise ValueError("worker timeout must be positive")

    started_wall = _utc_now()
    processes: dict[str, subprocess.Popen[str]] = {}
    records: dict[str, dict] = {}
    try:
        for role in ROLE_ORDER:
            start_mono = time.monotonic()
            proc = subprocess.Popen(
                commands[role],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes[role] = proc
            records[role] = {
                "pid": proc.pid,
                "started_at": _utc_now(),
                "_start_mono": start_mono,
            }
    except Exception:
        for proc in processes.values():
            if proc.poll() is None:
                proc.kill()
            proc.communicate()
        raise

    deadline = time.monotonic() + timeout_seconds
    pending = set(ROLE_ORDER)
    timed_out: list[str] = []
    failed_early: str | None = None
    while pending:
        now = time.monotonic()
        for role in tuple(pending):
            proc = processes[role]
            if proc.poll() is not None:
                records[role]["_end_mono"] = now
                records[role]["completed_at"] = _utc_now()
                pending.remove(role)
                if proc.returncode != 0:
                    failed_early = role
                    for other in pending:
                        processes[other].kill()
                        records[other]["_end_mono"] = now
                        records[other]["completed_at"] = _utc_now()
                    pending.clear()
                    break
        if not pending:
            break
        if now >= deadline:
            timed_out = sorted(pending)
            for role in pending:
                processes[role].kill()
                records[role]["_end_mono"] = now
                records[role]["completed_at"] = _utc_now()
            break
        time.sleep(0.01)

    for role in ROLE_ORDER:
        stdout, stderr = processes[role].communicate()
        record = records[role]
        record["returncode"] = processes[role].returncode
        record["stdout"] = stdout.strip()
        record["stderr"] = stderr.strip()
        record["duration_ms"] = round(
            1000 * (record["_end_mono"] - record["_start_mono"]), 3
        )

    overlap_ms = max(
        0.0,
        1000
        * (
            min(records[role]["_end_mono"] for role in ROLE_ORDER)
            - max(records[role]["_start_mono"] for role in ROLE_ORDER)
        ),
    )
    telemetry = {
        "schema_version": "1.0",
        "mode": "parallel-subprocess-fanout",
        "started_at": started_wall,
        "completed_at": _utc_now(),
        "timeout_seconds": timeout_seconds,
        "overlap_ms": round(overlap_ms, 3),
        "cancelled_after_failure": failed_early,
        "roles": {
            role: {
                key: value
                for key, value in records[role].items()
                if not key.startswith("_")
            }
            for role in ROLE_ORDER
        },
    }
    _write_telemetry(telemetry_path, telemetry)

    failures = [
        role for role in ROLE_ORDER if records[role]["returncode"] != 0
    ]
    if timed_out:
        raise TimeoutError(f"verification workers timed out: {timed_out}")
    if failures:
        details = "; ".join(
            f"{role} rc={records[role]['returncode']} stderr={records[role]['stderr']!r}"
            for role in failures
        )
        raise RuntimeError(f"verification worker failure: {details}")
    return telemetry["roles"]


def _load_role_results(
    path: Path,
    *,
    role: AgentRole,
    requests: list[VerificationCheckRequest],
) -> dict[str, ObjectPresenceCheckResult | PlacementComplianceResult]:
    expected_type = (
        ObjectPresenceCheckResult if role == AgentRole.MEM else PlacementComplianceResult
    )
    rows = load_trace(path)
    if any(not isinstance(row, expected_type) for row in rows):
        raise ValueError(f"{path}: contains messages outside {expected_type.__name__}")
    by_request: dict[str, ObjectPresenceCheckResult | PlacementComplianceResult] = {}
    for row in rows:
        if row.producer != role:
            raise ValueError(f"{row.message_id}: producer must be {role.value}")
        if row.request_id in by_request:
            raise ValueError(f"{path}: duplicate result for {row.request_id}")
        by_request[row.request_id] = row
    expected_ids = {request.message_id for request in requests}
    if set(by_request) != expected_ids:
        raise ValueError(
            f"{path}: result coverage mismatch; expected={sorted(expected_ids)} "
            f"actual={sorted(by_request)}"
        )
    return by_request


def _fan_in(
    cards: list[TaskCard],
    acceptance: AcceptanceManifest,
    requests: list[VerificationCheckRequest],
    mem_results: dict[str, ObjectPresenceCheckResult | PlacementComplianceResult],
    space_results: dict[str, ObjectPresenceCheckResult | PlacementComplianceResult],
) -> list[CardVerification]:
    cards_by_id = {card.card_id: card for card in cards}
    completed: list[CardVerification] = []
    for request in requests:
        presence = mem_results[request.message_id]
        compliance = space_results[request.message_id]
        if not isinstance(presence, ObjectPresenceCheckResult):
            raise TypeError(f"{request.message_id}: MEM returned wrong message type")
        if not isinstance(compliance, PlacementComplianceResult):
            raise TypeError(f"{request.message_id}: SPACE returned wrong message type")
        completed.append(
            finalize_verification(
                cards_by_id[request.task_id],
                request,
                presence,
                compliance,
                acceptance,
            )
        )
    return completed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", required=True, type=Path)
    ap.add_argument("--photos", required=True, type=Path)
    ap.add_argument("--photo-root", type=Path, default=PROJ)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace-parent", type=Path)
    ap.add_argument("--trace-out", type=Path)
    ap.add_argument("--worker-timeout-seconds", type=float, default=60.0)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    requests_path = out / "requests.jsonl"
    mem_path = out / "mem-results.jsonl"
    space_path = out / "space-results.jsonl"
    telemetry_path = out / "fanout-run.json"
    final_paths = [
        args.trace_out or (out / "messages.jsonl"),
        out / "messages.jsonl",
        out / "verdicts.json",
        out / "taskcards_verified.jsonl",
    ]
    # 先清理本阶段上一次运行的最终结论；后续任何输入/worker/协议失败都不得
    # 留下可被误认成当前成功结果的 stale verdict 或 combined trace。
    for path in {requests_path, mem_path, space_path, telemetry_path, *final_paths}:
        path.unlink(missing_ok=True)

    cards = _load_cards(args.cards)
    acceptance = AcceptanceManifest.model_validate_json(
        args.photos.read_text(encoding="utf-8")
    )
    selected_cards = _select_cards(cards, acceptance)
    parent = (
        require_handoff(args.trace_parent, "TASKS_READY")
        if args.trace_parent
        else None
    )
    requests = [
        build_verification_request(
            card,
            acceptance,
            parent_message_id=parent.message_id if parent else None,
        )
        for card in selected_cards
    ]
    _validate_request_photos(requests, args.photo_root)

    write_fragment(requests_path, requests)

    worker = PROJ / "scripts/verification_worker.py"
    base = [
        "--requests",
        str(requests_path),
        "--cards",
        str(args.cards),
        "--photos",
        str(args.photos),
    ]
    commands = {
        "MEM": [
            sys.executable,
            str(worker),
            "--role",
            "MEM",
            *base,
            "--out",
            str(mem_path),
        ],
        "SPACE": [
            sys.executable,
            str(worker),
            "--role",
            "SPACE",
            *base,
            "--out",
            str(space_path),
        ],
    }
    run_parallel_workers(
        commands,
        timeout_seconds=args.worker_timeout_seconds,
        telemetry_path=telemetry_path,
    )

    mem_results = _load_role_results(
        mem_path, role=AgentRole.MEM, requests=requests
    )
    space_results = _load_role_results(
        space_path, role=AgentRole.SPACE, requests=requests
    )
    results = _fan_in(
        selected_cards, acceptance, requests, mem_results, space_results
    )

    messages = []
    for result in results:
        messages.extend(
            [result.request, result.presence, result.compliance, result.verdict]
        )
        if result.adjudication:
            messages.append(result.adjudication)
    trace_out = args.trace_out or (out / "messages.jsonl")
    write_fragment(trace_out, messages)
    if trace_out != out / "messages.jsonl":
        write_fragment(out / "messages.jsonl", messages)

    summary = {
        result.card.card_id: {
            "verdict": result.verdict.verdict,
            "reason_codes": result.verdict.reason_codes,
            "photo_refs": result.request.photo_refs,
            "status_after": card_status_after(result).value,
            "adjudication": (
                {
                    "decision": result.adjudication.decision,
                    "note": result.adjudication.note,
                }
                if result.adjudication
                else None
            ),
        }
        for result in results
    }
    (out / "verdicts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_by_card = {result.card.card_id: result for result in results}
    with (out / "taskcards_verified.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards:
            result = result_by_card.get(card.card_id)
            status = card_status_after(result) if result else card.status
            verified = card.model_copy(update={"status": status})
            handle.write(verified.model_dump_json() + "\n")

    counts: dict[str, int] = {}
    for result in results:
        counts[result.verdict.verdict] = counts.get(result.verdict.verdict, 0) + 1
    print(
        json.dumps(
            {
                "cards_total": len(cards),
                "cards_selected": len(results),
                **counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
