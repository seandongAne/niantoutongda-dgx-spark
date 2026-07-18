from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from backend.schemas.hero_bundle import RegionManifest
from backend.tools.spatial.scoring import (
    FrozenSpatialTruthManifest,
    SemanticMatchStatus,
    score_spatial_regions,
    write_spatial_scoring_outputs,
)
from scripts.space_score_task import main as score_task_main


TRUTH_ROWS = [
    ("study_desk", "surface", "large", True),
    ("wall_shelf", "shelf", "medium", None),
    ("vanity_top", "surface", "medium", False),
    ("chest_top", "surface", "medium", None),
    ("display_cabinet", "shelf", "large", None),
]


def _truth() -> dict:
    return {
        "entries": [
            {
                "anchor": anchor,
                "support_type": support,
                "capacity_class": capacity,
                **({"near_power": power} if power is not None else {}),
            }
            for anchor, support, capacity, power in TRUTH_ROWS
        ]
    }


def _regions(*, reversed_order: bool = False) -> dict:
    rows = list(TRUTH_ROWS)
    if reversed_order:
        rows.reverse()
    return {
        "video_id": "new_1",
        "notes": "automatic producer output",
        "entries": [
            {
                "region_id": f"auto-{index}",
                "anchor": anchor.replace("_", "-").upper(),
                "display_name_zh": anchor,
                "support_type": support,
                "capacity_class": capacity,
                # Deliberately disagree where truth supplies power: power is
                # informational and must not change semantic acceptance.
                "near_power": not power if power is not None else False,
                "evidence_refs": [f"auto-frame-{index}.jpg"],
            }
            for index, (anchor, support, capacity, power) in enumerate(rows)
        ],
    }


def test_exact_five_of_five_is_order_invariant_and_ignores_power():
    first = score_spatial_regions(_regions(), _truth())
    reordered = score_spatial_regions(_regions(reversed_order=True), _truth())

    assert first.metrics.acceptance_passed
    assert first.metrics.score == "5/5"
    assert first.metrics.exact_semantic_match_count == 5
    assert first.metrics.missing_anchor_count == 0
    assert first.metrics.extra_prediction_count == 0
    assert first.metrics.informational_power_mismatch_count == 2
    assert first.normalized_hash == reordered.normalized_hash
    assert all(
        match.status is SemanticMatchStatus.EXACT
        for match in first.score_manifest.matches
    )
    serialized = first.score_manifest.model_dump_json()
    assert "region_id" not in serialized
    assert "track" not in serialized


def test_missing_extra_and_semantic_mismatches_fail_closed():
    prediction = _regions()
    prediction["entries"] = [
        entry for entry in prediction["entries"] if entry["anchor"] != "STUDY-DESK"
    ]
    prediction["entries"].append(
        {
            "region_id": "auto-extra",
            "anchor": "window-corner",
            "display_name_zh": "额外角落",
            "support_type": "floor",
            "capacity_class": "small",
            "near_power": False,
            "evidence_refs": ["extra.jpg"],
        }
    )
    for entry in prediction["entries"]:
        if entry["anchor"] == "WALL-SHELF":
            entry["capacity_class"] = "small"
        if entry["anchor"] == "VANITY-TOP":
            entry["support_type"] = "drawer"

    result = score_spatial_regions(prediction, _truth())

    assert not result.metrics.acceptance_passed
    assert result.metrics.score == "2/5"
    assert result.metrics.missing_anchor_count == 1
    assert result.metrics.extra_prediction_count == 1
    assert result.metrics.support_type_mismatch_count == 1
    assert result.metrics.capacity_class_mismatch_count == 1
    assert [match.anchor for match in result.score_manifest.matches] == sorted(
        row[0] for row in TRUTH_ROWS
    )
    assert result.score_manifest.extra_predictions[0].anchor == "window_corner"
    assert result.metrics.gate_reasons == [
        "missing_expected_anchors:1",
        "unexpected_prediction_anchors:1",
        "support_type_mismatches:1",
        "capacity_class_mismatches:1",
    ]


def test_duplicate_prediction_is_an_extra_and_cannot_pass():
    prediction = _regions()
    duplicate = copy.deepcopy(prediction["entries"][0])
    duplicate["region_id"] = "auto-duplicate"
    duplicate["support_type"] = "floor"
    prediction["entries"].append(duplicate)

    result = score_spatial_regions(prediction, _truth())

    assert result.metrics.score == "5/5"
    assert not result.metrics.acceptance_passed
    assert result.metrics.extra_prediction_count == 1
    assert result.metrics.gate_reasons == ["duplicate_prediction_anchors:1"]


@pytest.mark.parametrize(
    "forbidden",
    [
        {"region_id": "region-1"},
        {"candidate_id": "candidate-1"},
        {"track_id": "track-1"},
        {"source_spatial_normalized_hash": "a" * 64},
        {"relabel_operation": "KEEP_LABEL"},
    ],
)
def test_truth_contract_rejects_prediction_provenance(forbidden):
    payload = _truth()
    payload["entries"][0].update(forbidden)

    with pytest.raises(ValidationError):
        FrozenSpatialTruthManifest.model_validate(payload)


def test_truth_requires_five_unique_canonical_anchors_for_acceptance():
    four = _truth()
    four["entries"].pop()
    result = score_spatial_regions(
        {**_regions(), "entries": _regions()["entries"][:-1]},
        four,
    )
    assert not result.metrics.acceptance_passed
    assert result.metrics.score == "4/4"
    assert result.metrics.gate_reasons == [
        "truth_expected_anchor_count_mismatch:4!=5"
    ]

    duplicate = _truth()
    duplicate["entries"][1]["anchor"] = "STUDY DESK"
    with pytest.raises(ValidationError, match="truth anchors must be unique"):
        FrozenSpatialTruthManifest.model_validate(duplicate)


def test_writer_and_cli_emit_scorer_artifacts_but_never_regions(tmp_path):
    result = score_spatial_regions(_regions(), _truth())
    direct_dir = tmp_path / "direct"
    outputs = write_spatial_scoring_outputs(result, direct_dir)
    assert set(outputs) == {"score_manifest", "metrics", "normalized_hash"}
    assert sorted(path.name for path in direct_dir.iterdir()) == [
        "metrics.json",
        "normalized.sha256",
        "score_manifest.json",
    ]
    assert not (direct_dir / "regions.json").exists()

    regions_path = tmp_path / "auto-regions.json"
    truth_path = tmp_path / "truth.json"
    regions_path.write_text(json.dumps(_regions()), encoding="utf-8")
    truth_path.write_text(json.dumps(_truth()), encoding="utf-8")
    cli_dir = tmp_path / "cli"
    exit_code = score_task_main(
        [
            "--regions",
            str(regions_path),
            "--truth",
            str(truth_path),
            "--out-dir",
            str(cli_dir),
        ]
    )

    assert exit_code == 0
    assert not (cli_dir / "regions.json").exists()
    metrics = json.loads((cli_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["score"] == "5/5"
    assert metrics["acceptance_passed"] is True


def test_prediction_contract_requires_support_and_capacity():
    payload = _regions()
    payload["entries"][0].pop("support_type")
    with pytest.raises(ValidationError):
        RegionManifest.model_validate(payload)
