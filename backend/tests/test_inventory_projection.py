"""Tests for the label-independent hero inventory projection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.schemas.core import ObjectEntity
from backend.tools.inventory import (
    build_inventory_projection,
    project_inventory_files,
    write_inventory_projection,
)

ROOT = Path(__file__).resolve().parents[2]


def _synthetic_inputs() -> tuple[list[dict], dict, dict]:
    items = []
    anchors = []
    entities = []
    grouped_ids = {8: "pen", 18: "tea_bag", 19: "book"}
    for index in range(1, 21):
        canonical_id = grouped_ids.get(index, f"item_{index:02d}")
        tracklet_id = f"v1_t{index:02d}"
        items.append(
            {
                "canonical_id": canonical_id,
                "name_zh": f"物品{index:02d}",
                "description_zh": "synthetic",
            }
        )
        anchors.append(
            {
                "anchor_id": f"anchor_{index:02d}",
                "category_id": canonical_id,
                "display_label_zh": f"锚点{index:02d}",
                "visible_in": ["v1"],
                "confirmed_tracklet_ids_by_video": {"v1": [tracklet_id]},
            }
        )
        entities.append(
            {
                "entity_id": f"entity_{index:02d}",
                # Deliberately wrong: neither trusted identity nor raw-link
                # scoring may consult this label.
                "label": "wrong_label_for_every_entity",
                "identity_state": "MATCHED",
                "confidence": 0.99,
                "tracklet_ids": [tracklet_id],
                "evidence_refs": [f"{tracklet_id}.jpg"],
            }
        )
    return entities, {"items": items}, {
        "status": "data_owner_confirmed",
        "entities": anchors,
    }


def _write_inputs(root: Path) -> dict[str, Path]:
    entities, items, review = _synthetic_inputs()
    entities_path = root / "entities.jsonl"
    items_path = root / "items.json"
    review_path = root / "review.json"
    entities_path.write_text(
        "".join(json.dumps(row) + "\n" for row in entities), encoding="utf-8"
    )
    items_path.write_text(json.dumps(items), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return {
        "entities_path": entities_path,
        "items_path": items_path,
        "anchor_review_path": review_path,
    }


def test_projection_uses_confirmed_tracks_not_entity_labels():
    entities, items, review = _synthetic_inputs()
    projection = build_inventory_projection(entities, items, review)

    assert len(projection.inventory) == 20
    assert len(projection.trusted_entities) == 20
    assert len(projection.display) == 20
    assert projection.metrics["status_counts"] == {"TRUSTED": 20}
    assert projection.metrics["raw_link_status_counts"] == {
        "CONFIRMED": 20,
        "PARTIAL": 0,
        "AMBIGUOUS": 0,
        "CONTAMINATED": 0,
        "MISSING": 0,
    }
    assert projection.metrics["label_matching_used"] is False

    first = projection.inventory[0]
    assert first["canonical_id"] == "item_01"
    assert first["entity_id"] == "hero_item_01"
    assert first["tracklet_ids"] == ["v1_t01"]
    assert first["entity"]["entity_id"] == "hero_item_01"
    assert first["entity"]["tracklet_ids"] == ["v1_t01"]
    assert first["raw_link"]["candidate_entity_id"] == "entity_01"
    assert first["status"] == "TRUSTED"
    assert first["downstream_eligible"] is True
    assert projection.display[0]["display_name_zh"] == "物品01"
    assert all(ObjectEntity.model_validate(row) for row in projection.trusted_entities)
    assert all(
        any(
            Path(ref).stem == tracklet_id
            or Path(ref).stem.startswith(f"{tracklet_id}_")
            for tracklet_id in entity["tracklet_ids"]
        )
        for entity in projection.trusted_entities
        for ref in entity["evidence_refs"]
    )


def test_raw_link_gaps_never_remove_confirmed_inventory_when_cap_is_exhausted():
    entities, items, review = _synthetic_inputs()
    # Missing raw candidate for anchor 01; the confirmed tracklet still belongs
    # to the stable trusted entity.
    entities = [row for row in entities if row["entity_id"] != "entity_01"]
    # Cross-anchor raw pollution for 02 and 03.
    by_id = {row["entity_id"]: row for row in entities}
    by_id["entity_02"]["tracklet_ids"].append("v1_t03")
    by_id["entity_03"]["tracklet_ids"].append("v1_t02")
    # Split two-video raw coverage for anchors 04, 05, 06.
    for index in (4, 5, 6):
        anchor = review["entities"][index - 1]
        anchor["visible_in"].append("v2")
        anchor["confirmed_tracklet_ids_by_video"]["v2"] = [f"v2_t{index:02d}"]
        entities.append(
            {
                "entity_id": f"entity_{index:02d}_fragment",
                "label": "also_wrong",
                "identity_state": "NEW_ENTITY",
                "confidence": 0.8,
                "tracklet_ids": [f"v2_t{index:02d}"],
                "evidence_refs": [],
            }
        )

    projection = build_inventory_projection(entities, items, review)
    unresolved = [
        row
        for row in projection.inventory
        if row["raw_link"]["status"] != "CONFIRMED"
    ]

    assert len(unresolved) == 6
    assert len(projection.clarifications) == 4
    assert projection.metrics["deferred_unresolved_count"] == 2
    assert projection.metrics["trusted_inventory_count"] == 20
    assert projection.metrics["downstream_eligible_count"] == 20
    assert all(row["status"] == "TRUSTED" for row in projection.inventory)
    assert all(row["downstream_eligible"] for row in projection.inventory)
    assert all(
        row["clarification"]["state"] in {"SELECTED", "DEFERRED_BY_CAP"}
        for row in unresolved
    )
    missing = next(row for row in projection.inventory if row["anchor_id"] == "anchor_01")
    assert missing["tracklet_ids"] == ["v1_t01"]
    assert missing["raw_link"]["status"] == "MISSING"
    assert all(projection.metrics["gates"].values())


def test_projection_is_deterministic_and_keeps_group_quantity(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    first = project_inventory_files(**paths)
    second = project_inventory_files(**paths)
    assert first.projection_hash == second.projection_hash
    assert len(first.inventory) == 20
    assert len(first.clarifications) <= 4
    assert all(first.metrics["gates"].values())

    by_canonical = {row["canonical_id"]: row for row in first.inventory}
    assert by_canonical["pen"]["quantity"] == 6
    assert by_canonical["tea_bag"]["quantity"] == 2
    assert by_canonical["book"]["quantity"] == 3
    assert len([row for row in first.inventory if row["quantity"] > 1]) == 3

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    write_inventory_projection(first, out_a)
    write_inventory_projection(second, out_b)
    for name in (
        "inventory.jsonl",
        "trusted_entities.jsonl",
        "display.jsonl",
        "clarifications.jsonl",
        "metrics.json",
        "manifest.json",
        "hashes.json",
    ):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_inventory_cli_writes_all_audit_artifacts(tmp_path: Path):
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "projection"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/inventory_task.py",
            "--entities",
            str(paths["entities_path"]),
            "--items",
            str(paths["items_path"]),
            "--anchor-review",
            str(paths["anchor_review_path"]),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["raw_entities"] == 20
    assert summary["trusted_inventory"] == 20
    assert summary["downstream_eligible"] == 20
    assert summary["clarifications"] <= 4
    assert {path.name for path in out_dir.iterdir()} == {
        "inventory.jsonl",
        "trusted_entities.jsonl",
        "display.jsonl",
        "clarifications.jsonl",
        "metrics.json",
        "manifest.json",
        "hashes.json",
    }
