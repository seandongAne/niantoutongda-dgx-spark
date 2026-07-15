"""追加式审计事件存储 — JSONL,禁止覆盖旧事件(设计文档 §7.3)。"""

from __future__ import annotations

import json
from pathlib import Path

from backend.schemas.core import AuditEvent


def append_event(path: str | Path, event: AuditEvent) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")


def replay(path: str | Path) -> list[AuditEvent]:
    p = Path(path)
    if not p.exists():
        return []
    events: list[AuditEvent] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(AuditEvent.model_validate(json.loads(line)))
    return events
