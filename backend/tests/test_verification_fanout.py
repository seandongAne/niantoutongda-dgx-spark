"""验收阶段的双进程 fan-out/fan-in、范围与 fail-closed 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from backend.schemas.hero_bundle import (
    AcceptanceManifest,
    AcceptanceMatch,
    AcceptancePhoto,
    TaskCard,
    TaskCardItem,
)
from scripts.verify_task import run_parallel_workers
from scripts.hero_pipeline import build_stages

PROJ = Path(__file__).resolve().parent.parent.parent


def test_pipeline_tracks_worker_code_role_fragments_and_photo_bytes():
    config_path = PROJ / "configs/hero_pipeline_dev.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    stage = build_stages(config, sys.executable, config_path=config_path)["verify"]

    assert PROJ / "fixtures/hero_dev/acceptance/desk_after.ppm" in stage.inputs
    assert PROJ / "fixtures/hero_dev/acceptance/closet_after.ppm" in stage.inputs
    assert PROJ / "fixtures/hero_dev/acceptance/bedside_after.ppm" in stage.inputs
    assert PROJ / "scripts/verification_worker.py" in stage.code_paths
    assert PROJ / "backend/tools/verification/acceptance.py" in stage.code_paths
    assert {path.name for path in stage.outputs} >= {
        "requests.jsonl",
        "mem-results.jsonl",
        "space-results.jsonl",
    }


def _card(card_id: str, entity_id: str, region_id: str) -> TaskCard:
    return TaskCard(
        card_id=card_id,
        group_id=f"group-{card_id}",
        box_label_zh=f"{card_id} 测试箱",
        items=[TaskCardItem(entity_id=entity_id, display_name_zh=entity_id)],
        target_region_id=region_id,
        target_region_name_zh=region_id,
    )


def _write_cards(path: Path, cards: list[TaskCard]) -> None:
    path.write_text(
        "".join(card.model_dump_json() + "\n" for card in cards),
        encoding="utf-8",
    )


def _run_verify(cards: Path, acceptance: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PROJ / "scripts/verify_task.py"),
            "--cards",
            str(cards),
            "--photos",
            str(acceptance),
            "--out-dir",
            str(out),
            "--worker-timeout-seconds",
            "5",
        ],
        cwd=PROJ,
        capture_output=True,
        text=True,
    )


def test_selected_scope_only_verifies_selected_card_and_preserves_others(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    cards = [
        _card("card-01", "entity-01", "desk"),
        _card("card-02", "entity-02", "shelf"),
    ]
    _write_cards(cards_path, cards)
    photo = tmp_path / "desk-after.ppm"
    photo.write_text("P3\n1 1\n255\n1 2 3\n", encoding="ascii")
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        AcceptanceManifest(
            selected_card_ids=["card-01"],
            photos=[
                AcceptancePhoto(
                    photo_ref=str(photo),
                    region_id="desk",
                    matches=[
                        AcceptanceMatch(
                            entity_id="entity-01", present=True, match_score=0.9
                        )
                    ],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )

    first_out = tmp_path / "verify-first"
    first = _run_verify(cards_path, acceptance_path, first_out)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == {
        "VERIFIED": 1,
        "cards_selected": 1,
        "cards_total": 2,
    }
    verdicts = json.loads((first_out / "verdicts.json").read_text(encoding="utf-8"))
    assert set(verdicts) == {"card-01"}
    updated = [
        json.loads(line)
        for line in (first_out / "taskcards_verified.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["card_id"] for row in updated] == ["card-01", "card-02"]
    assert [row["status"] for row in updated] == ["VERIFIED", "REGION_PLANNED"]

    telemetry = json.loads((first_out / "fanout-run.json").read_text(encoding="utf-8"))
    assert telemetry["mode"] == "parallel-subprocess-fanout"
    assert set(telemetry["roles"]) == {"MEM", "SPACE"}
    assert all(telemetry["roles"][role]["returncode"] == 0 for role in ("MEM", "SPACE"))

    second_out = tmp_path / "verify-second"
    second = _run_verify(cards_path, acceptance_path, second_out)
    assert second.returncode == 0, second.stderr
    for name in (
        "requests.jsonl",
        "mem-results.jsonl",
        "space-results.jsonl",
        "messages.jsonl",
        "verdicts.json",
        "taskcards_verified.jsonl",
    ):
        assert (first_out / name).read_bytes() == (second_out / name).read_bytes()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_photo_fails_closed_and_removes_stale_results(tmp_path, empty):
    cards_path = tmp_path / "cards.jsonl"
    _write_cards(cards_path, [_card("card-01", "entity-01", "desk")])
    photo = tmp_path / "missing-or-empty.ppm"
    if empty:
        photo.write_bytes(b"")
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        AcceptanceManifest(
            selected_card_ids=["card-01"],
            photos=[
                AcceptancePhoto(
                    photo_ref=str(photo),
                    region_id="desk",
                    matches=[AcceptanceMatch(entity_id="entity-01", present=True)],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    out = tmp_path / "verify"
    out.mkdir()
    for name in ("messages.jsonl", "verdicts.json", "taskcards_verified.jsonl"):
        (out / name).write_text("stale", encoding="utf-8")

    proc = _run_verify(cards_path, acceptance_path, out)

    assert proc.returncode != 0
    for name in ("messages.jsonl", "verdicts.json", "taskcards_verified.jsonl"):
        assert not (out / name).exists()
    assert not (out / "requests.jsonl").exists()


def test_selected_card_without_relevant_photo_fails_before_workers(tmp_path):
    cards_path = tmp_path / "cards.jsonl"
    _write_cards(cards_path, [_card("card-01", "entity-01", "desk")])
    photo = tmp_path / "other.ppm"
    photo.write_text("P3\n1 1\n255\n1 2 3\n", encoding="ascii")
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        AcceptanceManifest(
            selected_card_ids=["card-01"],
            photos=[
                AcceptancePhoto(
                    photo_ref=str(photo),
                    region_id="shelf",
                    matches=[AcceptanceMatch(entity_id="other", present=True)],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    out = tmp_path / "verify"

    proc = _run_verify(cards_path, acceptance_path, out)

    assert proc.returncode != 0
    assert "no relevant verification photos" in proc.stderr
    assert not (out / "fanout-run.json").exists()
    assert not (out / "messages.jsonl").exists()


def test_worker_processes_overlap_and_single_branch_failure_is_fatal(tmp_path):
    sleeper = "import time; time.sleep(0.2)"
    telemetry = tmp_path / "parallel.json"
    roles = run_parallel_workers(
        {
            "MEM": [sys.executable, "-c", sleeper],
            "SPACE": [sys.executable, "-c", sleeper],
        },
        timeout_seconds=2,
        telemetry_path=telemetry,
    )
    report = json.loads(telemetry.read_text(encoding="utf-8"))
    assert report["overlap_ms"] >= 100
    assert roles["MEM"]["returncode"] == roles["SPACE"]["returncode"] == 0

    failed_telemetry = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="MEM rc=7"):
        run_parallel_workers(
            {
                "MEM": [sys.executable, "-c", "raise SystemExit(7)"],
                "SPACE": [sys.executable, "-c", "import time; time.sleep(2)"],
            },
            timeout_seconds=2,
            telemetry_path=failed_telemetry,
        )
    failed = json.loads(failed_telemetry.read_text(encoding="utf-8"))
    assert failed["cancelled_after_failure"] == "MEM"
    assert failed["roles"]["MEM"]["returncode"] == 7
    assert failed["roles"]["SPACE"]["returncode"] != 0


def test_worker_timeout_kills_both_branches(tmp_path):
    telemetry = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError, match="timed out"):
        run_parallel_workers(
            {
                "MEM": [sys.executable, "-c", "import time; time.sleep(2)"],
                "SPACE": [sys.executable, "-c", "import time; time.sleep(2)"],
            },
            timeout_seconds=0.05,
            telemetry_path=telemetry,
        )
    report = json.loads(telemetry.read_text(encoding="utf-8"))
    assert all(report["roles"][role]["returncode"] != 0 for role in ("MEM", "SPACE"))
