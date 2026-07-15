from backend.schemas.core import AuditEvent
from backend.tools.audit.store import append_event, replay


def _ev(i: int) -> AuditEvent:
    return AuditEvent(
        event_id=f"ev{i}",
        event_type="EntityResolved",
        actor="object_memory",
        input_refs=[f"t{i}"],
        output_refs=[f"e{i}"],
        config_version="reid-v0",
        created_at=f"2026-07-15T12:00:0{i}Z",
    )


def test_append_and_replay_order(tmp_path):
    path = tmp_path / "audit" / "events.jsonl"
    for i in range(3):
        append_event(path, _ev(i))
    events = replay(path)
    assert [e.event_id for e in events] == ["ev0", "ev1", "ev2"]


def test_append_never_truncates(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, _ev(0))
    first = path.read_text()
    append_event(path, _ev(1))
    assert path.read_text().startswith(first)  # 旧事件原样保留
