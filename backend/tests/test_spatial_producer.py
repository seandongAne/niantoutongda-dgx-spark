import json

import pytest
from pydantic import ValidationError

from scripts.space_task import build_parser, main as space_task_main

from backend.tools.spatial import (
    CandidateStatus,
    CoverageStatus,
    GateStatus,
    PowerState,
    SpatialObservation,
    SpatialProducerConfig,
    load_observations_jsonl,
    produce_spatial_regions,
    write_spatial_outputs,
)


def _observation(
    anchor: str,
    frame: int,
    bbox: list[float],
    *,
    support: str = "surface",
    capacity: str = "medium",
    confidence: float = 0.92,
    power: str = "UNKNOWN",
    power_evidence: list[str] | None = None,
    **extra,
) -> dict:
    value = {
        "timestamp": frame * 100,
        "frame": frame,
        "bbox": bbox,
        "anchor_label": anchor,
        "display_name_zh": anchor,
        "support": support,
        "capacity": capacity,
        "power": power,
        "power_evidence": power_evidence or [],
        "confidence": confidence,
        "model_version": "spatial-test-v1",
    }
    value.update(extra)
    return value


def _five_region_observations() -> list[dict]:
    anchors = ["bed", "desk", "closet", "shelf", "corner"]
    observations: list[dict] = []
    for index, anchor in enumerate(anchors):
        x1 = index * 0.12
        kwargs = {}
        if anchor == "desk":
            kwargs = {
                "power": "NEAR",
                "power_evidence": ["new_1@1200ms#outlet-det-4"],
                "power_confidence": 0.94,
            }
        observations.extend(
            [
                _observation(anchor, index * 2 + 1, [x1, 0.1, x1 + 0.1, 0.3], **kwargs),
                _observation(
                    anchor,
                    index * 2 + 2,
                    [x1 + 0.005, 0.105, x1 + 0.105, 0.305],
                    **kwargs,
                ),
            ]
        )
    return observations


def test_stable_cross_frame_dedup_and_region_manifest_projection():
    observations = _five_region_observations()
    config = SpatialProducerConfig(
        min_regions=5,
        expected_anchor_labels=["bed", "desk", "closet", "shelf", "corner"],
    )

    result = produce_spatial_regions("new_1", observations, config)
    reordered = produce_spatial_regions("new_1", list(reversed(observations)), config)

    assert result.gate_passed
    assert result.metrics.gate_status is GateStatus.PASS
    assert result.metrics.auto_accepted_count == 5
    assert result.metrics.projected_region_count == 5
    assert len(result.candidate_manifest.candidates) == 5
    assert len({item.region_id for item in result.candidate_manifest.candidates}) == 5
    assert all(
        item.status is CandidateStatus.AUTO_ACCEPTED
        for item in result.candidate_manifest.candidates
    )
    assert result.normalized_hash == reordered.normalized_hash
    assert [
        item.region_id for item in result.candidate_manifest.candidates
    ] == [item.region_id for item in reordered.candidate_manifest.candidates]

    assert result.region_manifest is not None
    desk = next(entry for entry in result.region_manifest.entries if entry.anchor == "desk")
    assert desk.near_power is True
    assert "new_1@1200ms#outlet-det-4" in desk.evidence_refs
    assert all(entry.region_id.startswith("auto_") for entry in result.region_manifest.entries)
    assert all(
        candidate.model_versions == ["spatial-test-v1"]
        for candidate in result.candidate_manifest.candidates
    )


def test_near_power_without_outlet_evidence_is_not_promoted():
    observations = [
        _observation("desk", 1, [0.1, 0.1, 0.5, 0.4], power="NEAR"),
        _observation("desk", 2, [0.11, 0.1, 0.51, 0.4], power="NEAR"),
    ]
    result = produce_spatial_regions(
        "new_1", observations, SpatialProducerConfig(min_regions=1)
    )

    candidate = result.candidate_manifest.candidates[0]
    assert candidate.power_state is PowerState.UNKNOWN
    assert candidate.status is CandidateStatus.NEEDS_USER
    assert "power_near_missing_or_low_confidence_outlet_evidence" in candidate.decision_reasons
    assert result.region_manifest is None
    assert result.metrics.region_gate_passed is False


def test_uncertain_hard_field_is_fail_closed():
    observations = [
        _observation(
            "closet",
            1,
            [0.1, 0.1, 0.4, 0.6],
            support="shelf",
            support_confidence=0.95,
        ),
        _observation(
            "closet",
            2,
            [0.11, 0.1, 0.41, 0.6],
            support="drawer",
            support_confidence=0.95,
        ),
    ]
    result = produce_spatial_regions(
        "new_1", observations, SpatialProducerConfig(min_regions=1)
    )

    candidate = result.candidate_manifest.candidates[0]
    assert candidate.support_type is None
    assert candidate.status is CandidateStatus.NEEDS_USER
    assert "support_type_uncertain" in candidate.decision_reasons
    assert result.region_manifest is None


def test_expected_anchor_not_observed_is_explicit_and_blocks_gate():
    observations = [
        _observation("bed", 1, [0.1, 0.1, 0.4, 0.4]),
        _observation("bed", 2, [0.11, 0.1, 0.41, 0.4]),
    ]
    config = SpatialProducerConfig(
        min_regions=1,
        expected_anchor_labels=["bed", "desk"],
    )
    result = produce_spatial_regions("new_1", observations, config)

    missing = next(
        item
        for item in result.candidate_manifest.candidates
        if item.coverage_status is CoverageStatus.NOT_OBSERVED
    )
    assert missing.anchor == "desk"
    assert missing.status is CandidateStatus.NEEDS_USER
    assert result.metrics.expected_coverage_rate == 0.5
    assert result.metrics.gate_status is GateStatus.NEEDS_USER
    assert result.region_manifest is None


def test_observed_but_unaccepted_expected_anchor_cannot_be_replaced_by_duplicate_regions():
    observations: list[dict] = []
    for index, anchor in enumerate(["bed", "bed", "desk", "closet", "shelf"]):
        x1 = index * 0.15
        observations.extend(
            [
                _observation(
                    anchor,
                    index * 3 + 1,
                    [x1, 0.1, x1 + 0.1, 0.3],
                    region_track_id=f"{anchor}-{index}",
                ),
                _observation(
                    anchor,
                    index * 3 + 2,
                    [x1 + 0.005, 0.105, x1 + 0.105, 0.305],
                    region_track_id=f"{anchor}-{index}",
                ),
            ]
        )
    observations.append(
        _observation(
            "corner",
            99,
            [0.82, 0.1, 0.92, 0.3],
            region_track_id="corner-0",
        )
    )

    result = produce_spatial_regions(
        "new_1",
        observations,
        SpatialProducerConfig(
            min_regions=5,
            expected_anchor_labels=["bed", "desk", "closet", "shelf", "corner"],
        ),
    )

    assert result.metrics.auto_accepted_count == 5
    assert result.metrics.expected_coverage_rate == 1.0
    assert result.metrics.gate_status is GateStatus.NEEDS_USER
    assert result.region_manifest is None
    assert result.metrics.gate_reasons == [
        "expected_anchors_not_auto_accepted:corner"
    ]


def test_polygon_input_aliases_and_jsonl_loader(tmp_path):
    path = tmp_path / "observations.jsonl"
    payload = {
        "video_id": "new_1",
        "timestamp_ms": 100,
        "frame_id": "f001",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "anchor": "floor_corner",
        "support_type": "floor",
        "capacity_class": "large",
        "model_confidence": 0.9,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    observations = load_observations_jsonl(path, video_id="new_1")
    assert observations[0].frame_ref == "f001"
    assert observations[0].polygon is not None

    with pytest.raises(ValueError, match="does not match"):
        load_observations_jsonl(path, video_id="another_video")


def test_spatial_observation_requires_bbox_or_polygon():
    with pytest.raises(ValidationError, match="one of bbox or polygon"):
        SpatialObservation.model_validate(
            {
                "timestamp_ms": 1,
                "frame_ref": "f1",
                "anchor_label": "desk",
                "support_type": "surface",
                "capacity_class": "medium",
                "model_confidence": 0.9,
            }
        )


def test_writer_removes_stale_region_manifest_on_failed_gate(tmp_path):
    passing = produce_spatial_regions(
        "new_1",
        _five_region_observations(),
        SpatialProducerConfig(min_regions=5),
    )
    write_spatial_outputs(passing, tmp_path)
    assert (tmp_path / "regions.json").exists()

    failing = produce_spatial_regions(
        "new_1",
        [_observation("bed", 1, [0.1, 0.1, 0.4, 0.4])],
        SpatialProducerConfig(min_regions=1),
    )
    written = write_spatial_outputs(failing, tmp_path)
    assert not (tmp_path / "regions.json").exists()
    assert set(written) == {"candidate_manifest", "metrics", "normalized_hash"}
    assert (tmp_path / "normalized.sha256").read_text().strip() == failing.normalized_hash


def test_cli_accepts_only_auto_observation_input_and_writes_four_outputs(tmp_path):
    observations_path = tmp_path / "auto-observations.jsonl"
    observations_path.write_text(
        "".join(json.dumps(item) + "\n" for item in _five_region_observations()),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    code = space_task_main(
        [
            "--video-id",
            "new_1",
            "--observations",
            str(observations_path),
            "--out-dir",
            str(out_dir),
            "--min-regions",
            "5",
        ]
    )

    assert code == 0
    assert {path.name for path in out_dir.iterdir()} == {
        "candidate_manifest.json",
        "regions.json",
        "metrics.json",
        "normalized.sha256",
    }


def test_cli_shadow_mode_keeps_failed_gate_as_diagnostics_only(tmp_path):
    observations_path = tmp_path / "auto-observations.jsonl"
    observations_path.write_text(
        json.dumps(_observation("bed", 1, [0.1, 0.1, 0.4, 0.4])) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "shadow"

    code = space_task_main(
        [
            "--video-id",
            "new_1",
            "--observations",
            str(observations_path),
            "--out-dir",
            str(out_dir),
            "--min-regions",
            "5",
            "--shadow-only",
        ]
    )

    assert code == 0
    assert {path.name for path in out_dir.iterdir()} == {
        "candidate_manifest.json",
        "metrics.json",
        "normalized.sha256",
    }
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gate_status"] == "NEEDS_USER"
    assert metrics["region_gate_passed"] is False
    assert "--manifest" not in build_parser().format_help()
