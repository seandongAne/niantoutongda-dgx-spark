"""验收照片 → 消息族切片(acceptance.py + verify_task.py)。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.schemas.core import TaskStatus
from backend.schemas.hero_bundle import (
    AcceptanceAdjudication,
    AcceptanceManifest,
    AcceptanceMatch,
    AcceptancePhoto,
    TaskCard,
    TaskCardItem,
)
from backend.tools.verification.acceptance import (
    card_status_after,
    verify_card,
)

PROJ = Path(__file__).resolve().parent.parent.parent


def _card(card_id: str = "card-01", entity_ids: tuple[str, ...] = ("e1", "e2")) -> TaskCard:
    return TaskCard(
        card_id=card_id,
        group_id="g01",
        box_label_zh="测试箱",
        items=[
            TaskCardItem(entity_id=eid, display_name_zh=f"物品{eid}")
            for eid in entity_ids
        ],
        target_region_id="desk_top",
        target_region_name_zh="书桌面",
    )


def _manifest(photos: list[AcceptancePhoto]) -> AcceptanceManifest:
    return AcceptanceManifest(photos=photos)


def _photo(region_id: str, matches: list[AcceptanceMatch], ref: str = "p1.jpg") -> AcceptancePhoto:
    return AcceptancePhoto(photo_ref=ref, region_id=region_id, matches=matches)


def test_all_present_in_target_region_verified():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
            AcceptanceMatch(entity_id="e2", present=True, match_score=0.8),
        ]),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "VERIFIED"
    assert card_status_after(r) == TaskStatus.VERIFIED


def test_missing_entity_fails_with_not_seen():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
            AcceptanceMatch(entity_id="e2", present=False),
        ]),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "FAILED"
    assert "NOT_SEEN:e2" in r.verdict.reason_codes
    # FAILED 不代写卡状态,留给用户裁决
    assert card_status_after(r) == TaskStatus.REGION_PLANNED


def test_wrong_region_fails_with_misplaced():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
        ]),
        _photo("bedside", [
            AcceptanceMatch(entity_id="e2", present=True, match_score=0.9),
        ], ref="p2.jpg"),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "FAILED"
    assert any(c.startswith("MISPLACED:e2:WRONG_REGION:bedside")
               for c in r.verdict.reason_codes)
    # 拍错区域的照片也是证据,必须进 photo_refs
    assert "p2.jpg" in r.request.photo_refs


def test_low_confidence_needs_user():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
            AcceptanceMatch(entity_id="e2", present=True, match_score=0.5),
        ]),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "NEEDS_USER"
    assert r.verdict.reason_codes == ["LOW_CONFIDENCE:e2"]
    assert card_status_after(r) == TaskStatus.REGION_PLANNED


def test_user_accept_override_is_standalone_message_and_updates_status():
    manifest = AcceptanceManifest(
        photos=[
            _photo("desk_top", [
                AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
                AcceptanceMatch(entity_id="e2", present=True, match_score=0.5),
            ])
        ],
        adjudications=[
            AcceptanceAdjudication(
                card_id="card-01",
                decision="accept_override",
                note="用户确认是同一件",
            )
        ],
    )
    result = verify_card(_card(), manifest)
    assert result.verdict.verdict == "NEEDS_USER"
    assert result.adjudication is not None
    assert result.adjudication.causation_id == result.verdict.message_id
    assert card_status_after(result) == TaskStatus.USER_OVERRIDDEN


def test_no_relevant_photo_all_not_seen():
    manifest = _manifest([
        _photo("closet_shelf", [
            AcceptanceMatch(entity_id="other", present=True, match_score=0.9),
        ]),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "FAILED"
    assert set(r.verdict.reason_codes) == {"NOT_SEEN:e1", "NOT_SEEN:e2"}


def test_best_match_across_photos_wins():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.55),
            AcceptanceMatch(entity_id="e2", present=True, match_score=0.9),
        ]),
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.92,
                            evidence_refs=["crop_a.jpg"]),
        ], ref="p2.jpg"),
    ])
    r = verify_card(_card(), manifest)
    assert r.verdict.verdict == "VERIFIED"
    e1 = next(p for p in r.presence.presences if p.entity_id == "e1")
    assert e1.match_score == 0.92
    assert e1.evidence_refs == ["p2.jpg", "crop_a.jpg"]


def test_message_chain_deterministic():
    manifest = _manifest([
        _photo("desk_top", [
            AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
            AcceptanceMatch(entity_id="e2", present=True, match_score=0.8),
        ]),
    ])
    a = verify_card(_card(), manifest)
    b = verify_card(_card(), manifest)
    for x, y in ((a.request, b.request), (a.presence, b.presence),
                 (a.compliance, b.compliance), (a.verdict, b.verdict)):
        assert x.payload_hash and x.payload_hash == y.payload_hash
    assert a.verdict.request_id == a.request.message_id
    assert a.verdict.correlation_id == "verify-card-01"


def test_duplicate_photo_ref_rejected():
    with pytest.raises(ValueError):
        _manifest([
            _photo("desk_top", []),
            _photo("bedside", []),
        ])


def test_cli_writes_messages_verdicts_and_updated_cards(tmp_path):
    cards_path = tmp_path / "taskcards.jsonl"
    cards_path.write_text(_card().model_dump_json() + "\n", encoding="utf-8")
    photos_path = tmp_path / "acceptance.json"
    photo_path = tmp_path / "p1.jpg"
    photo_path.write_bytes(b"synthetic-test-photo")
    photos_path.write_text(
        _manifest([
            _photo("desk_top", [
                AcceptanceMatch(entity_id="e1", present=True, match_score=0.9),
                AcceptanceMatch(entity_id="e2", present=True, match_score=0.8),
            ], ref=str(photo_path)),
        ]).model_dump_json(),
        encoding="utf-8",
    )
    out = tmp_path / "verify"
    proc = subprocess.run(
        [sys.executable, str(PROJ / "scripts/verify_task.py"),
         "--cards", str(cards_path), "--photos", str(photos_path),
         "--out-dir", str(out)],
        capture_output=True, text=True, cwd=PROJ,
    )
    assert proc.returncode == 0, proc.stderr
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    assert verdicts["card-01"]["verdict"] == "VERIFIED"
    assert verdicts["card-01"]["status_after"] == "VERIFIED"
    lines = (out / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["message_type"] for l in lines] == [
        "VerificationCheckRequest",
        "ObjectPresenceCheckResult",
        "PlacementComplianceResult",
        "VerificationVerdict",
    ]
    updated = json.loads(
        (out / "taskcards_verified.jsonl").read_text(encoding="utf-8")
    )
    assert updated["status"] == "VERIFIED"
