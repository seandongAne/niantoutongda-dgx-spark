import json
from pathlib import Path

from PIL import Image

from backend.schemas.core import Observation, Tracklet
from backend.tools.spatial import (
    PowerState,
    SpatialProducerConfig,
    load_observations_jsonl,
    produce_spatial_regions,
)
from scripts.space_observation_adapter import (
    ALLOWED_SPACE_CONCEPTS,
    AdapterConfig,
    adapt_space_observations,
    build_parser,
)

ROOT = Path(__file__).resolve().parents[2]
SPACE_VOCAB = ROOT / "fixtures" / "hero_s1" / "space_vocab.json"
FURNITURE = [
    "study_desk",
    "vanity",
    "wall_shelf",
    "chest_of_drawers",
    "display_cabinet",
]


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(value.model_dump_json() + "\n" for value in values),
        encoding="utf-8",
    )


def _build_ingest(
    root: Path,
    *,
    keyframes: bool,
    outlet_crop_ref: str = "",
) -> Path:
    root.mkdir(parents=True)
    if keyframes:
        keyframes_dir = root / "keyframes"
        keyframes_dir.mkdir()
        for frame_index in (0, 1):
            Image.new("RGB", (1000, 800), color=(240, 240, 240)).save(
                keyframes_dir / f"kf_{frame_index:06d}.jpg"
            )

    observations: list[Observation] = []
    tracklets: list[Tracklet] = []
    boxes = {
        "study_desk": (100.0, 100.0, 300.0, 300.0),
        "vanity": (350.0, 100.0, 500.0, 300.0),
        "wall_shelf": (550.0, 80.0, 750.0, 180.0),
        "chest_of_drawers": (100.0, 400.0, 300.0, 650.0),
        "display_cabinet": (600.0, 300.0, 850.0, 700.0),
    }
    detection_index = 0
    for concept_index, concept in enumerate(FURNITURE):
        observation_ids = []
        for frame_index in (0, 1):
            observation_id = f"new_1_f{frame_index:06d}_d{detection_index:02d}"
            detection_index += 1
            x1, y1, x2, y2 = boxes[concept]
            observation_ids.append(observation_id)
            observations.append(
                Observation(
                    observation_id=observation_id,
                    video_id="new_1",
                    timestamp_ms=frame_index * 500,
                    bbox=(x1 + frame_index, y1, x2 + frame_index, y2),
                    crop_ref=f"evidence/{concept}_{frame_index}.jpg",
                    quality=0.92 - concept_index * 0.01,
                    model_version="grounding-dino-space-test",
                )
            )
        label = "writing desk" if concept == "study_desk" else concept
        tracklets.append(
            Tracklet(
                tracklet_id=f"new_1_t{concept_index + 1:03d}",
                video_id="new_1",
                observation_ids=observation_ids,
                attributes={
                    "label": label,
                    "region_id": "manual-secret-region",
                    "anchor_id": "manual-secret-anchor",
                },
            )
        )

    outlet_id = f"new_1_f000000_d{detection_index:02d}"
    observations.append(
        Observation(
            observation_id=outlet_id,
            video_id="new_1",
            timestamp_ms=0,
            bbox=(305.0, 180.0, 325.0, 210.0),
            crop_ref=outlet_crop_ref,
            quality=0.91,
            model_version="grounding-dino-space-test",
        )
    )
    tracklets.append(
        Tracklet(
            tracklet_id="new_1_t900",
            video_id="new_1",
            observation_ids=[outlet_id],
            attributes={"label": "power socket"},
        )
    )
    _write_jsonl(root / "observations.jsonl", reversed(observations))
    _write_jsonl(root / "tracklets.jsonl", reversed(tracklets))
    return root


def _all_output_bytes(out_dir: Path) -> bytes:
    return b"".join(
        (out_dir / name).read_bytes()
        for name in ("auto_observations.jsonl", "metrics.json", "hashes.json")
    )


def test_space_vocab_is_inference_only_and_has_exact_concepts():
    raw = json.loads(SPACE_VOCAB.read_text(encoding="utf-8"))
    assert {entry["canonical_id"] for entry in raw["entries"]} == set(
        ALLOWED_SPACE_CONCEPTS
    )

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert not ({"region_id", "anchor_id"} & keys(raw))


def test_adapter_uses_pil_dimensions_near_evidence_and_never_emits_outlet_region(
    tmp_path,
):
    ingest_dir = _build_ingest(tmp_path / "ingest", keyframes=True)
    result = adapt_space_observations(
        ingest_dir=ingest_dir,
        vocab_path=SPACE_VOCAB,
        out_dir=tmp_path / "out",
        config=AdapterConfig(near_diagonal_ratio=0.15),
    )

    assert len(result.observations) == 10
    assert result.metrics["frame_size_source_counts"] == {"pil": 2}
    assert all(item.anchor_label != "electrical_outlet" for item in result.observations)
    desk = [item for item in result.observations if item.anchor_label == "study_desk"]
    assert [item.power_state for item in desk] == [PowerState.NEAR, PowerState.UNKNOWN]
    assert desk[0].power_evidence_refs
    assert any(ref.startswith("outlet_frame:") for ref in desk[0].power_evidence_refs)
    assert all(
        item.power_state is not PowerState.NOT_NEAR for item in result.observations
    )

    payloads = [
        json.loads(line)
        for line in (tmp_path / "out" / "auto_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("region_id" not in item and "anchor_id" not in item for item in payloads)
    all_outputs = _all_output_bytes(tmp_path / "out")
    assert b"manual-secret-region" not in all_outputs
    assert b"manual-secret-anchor" not in all_outputs


def test_missing_real_frame_or_outlet_crop_cannot_claim_near(tmp_path):
    ingest_dir = _build_ingest(tmp_path / "ingest", keyframes=False)
    result = adapt_space_observations(
        ingest_dir=ingest_dir,
        vocab_path=SPACE_VOCAB,
        out_dir=tmp_path / "out",
        config=AdapterConfig(
            near_diagonal_ratio=0.15,
            fallback_frame_size=(1000, 800),
        ),
    )
    desk = [item for item in result.observations if item.anchor_label == "study_desk"]
    assert all(item.power_state is PowerState.UNKNOWN for item in desk)
    assert result.metrics["frame_size_source_counts"] == {"explicit_fallback": 2}


def test_explicit_fallback_can_use_real_outlet_crop_reference(tmp_path):
    ingest_dir = _build_ingest(
        tmp_path / "ingest",
        keyframes=False,
        outlet_crop_ref="evidence/outlet.jpg",
    )
    result = adapt_space_observations(
        ingest_dir=ingest_dir,
        vocab_path=SPACE_VOCAB,
        out_dir=tmp_path / "out",
        config=AdapterConfig(
            near_diagonal_ratio=0.15,
            fallback_frame_size=(1000, 800),
        ),
    )
    desk = [item for item in result.observations if item.anchor_label == "study_desk"]
    assert desk[0].power_state is PowerState.NEAR
    assert any(ref.startswith("outlet_crop:") for ref in desk[0].power_evidence_refs)


def test_adapter_outputs_are_byte_stable_and_space_task_accepts_five_regions(tmp_path):
    ingest_dir = _build_ingest(tmp_path / "ingest", keyframes=True)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = adapt_space_observations(
        ingest_dir=ingest_dir,
        vocab_path=SPACE_VOCAB,
        out_dir=first_dir,
    )
    second = adapt_space_observations(
        ingest_dir=ingest_dir,
        vocab_path=SPACE_VOCAB,
        out_dir=second_dir,
    )
    assert _all_output_bytes(first_dir) == _all_output_bytes(second_dir)
    assert first.hashes["normalized_hash"] == second.hashes["normalized_hash"]

    auto = load_observations_jsonl(
        first_dir / "auto_observations.jsonl", video_id="new_1"
    )
    produced = produce_spatial_regions(
        "new_1",
        auto,
        SpatialProducerConfig(
            min_regions=5,
            expected_anchor_labels=FURNITURE,
        ),
    )
    assert produced.gate_passed
    assert produced.region_manifest is not None
    assert len(produced.region_manifest.entries) == 5


def test_cli_has_no_manual_manifest_input():
    assert "--manifest" not in build_parser().format_help()
