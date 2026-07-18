#!/usr/bin/env python
"""Adapt automatic new-home ingest artifacts to SpatialObservation JSONL.

Inputs are only the video's ``observations.jsonl``, ``tracklets.jsonl``,
keyframes, and the open-vocabulary inference vocabulary.  No hand-authored
region manifest is accepted or read.

Furniture tracklets become automatic region observations.  Electrical outlets
never become regions: an outlet may mark a furniture observation ``NEAR`` only
when both detections occur in the same frame, their bbox-center distance is
below the configured fraction of the image diagonal, and an outlet crop or
real keyframe reference exists.  Non-detection is always ``UNKNOWN`` rather
than evidence for ``NOT_NEAR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from PIL import Image
from pydantic import BaseModel

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.vocab import Vocabulary, load_vocabulary  # noqa: E402
from backend.schemas.core import Observation, Tracklet  # noqa: E402
from backend.tools.spatial import PowerState, SpatialObservation  # noqa: E402

ADAPTER_SCHEMA_VERSION = "1.0"
ADAPTER_VERSION = "space-observation-adapter-v1"

AUTO_OBSERVATIONS_FILENAME = "auto_observations.jsonl"
METRICS_FILENAME = "metrics.json"
HASHES_FILENAME = "hashes.json"

OUTLET_CONCEPT = "electrical_outlet"


@dataclass(frozen=True)
class RegionSpec:
    anchor_label: str
    support_type: str
    capacity_class: str
    display_name_zh: str


REGION_SPECS: Mapping[str, RegionSpec] = {
    "study_desk": RegionSpec("study_desk", "surface", "large", "学习桌面"),
    "vanity": RegionSpec("vanity", "surface", "medium", "梳妆台面"),
    "wall_shelf": RegionSpec("wall_shelf", "shelf", "medium", "墙面置物架"),
    "chest_of_drawers": RegionSpec(
        "chest_of_drawers", "surface", "medium", "斗柜台面"
    ),
    "display_cabinet": RegionSpec(
        "display_cabinet", "shelf", "large", "展示柜层板"
    ),
}
ALLOWED_SPACE_CONCEPTS = frozenset({*REGION_SPECS, OUTLET_CONCEPT})


@dataclass(frozen=True)
class AdapterConfig:
    near_diagonal_ratio: float = 0.15
    fallback_frame_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.near_diagonal_ratio < 1.0:
            raise ValueError("near_diagonal_ratio must be in (0, 1)")
        if self.fallback_frame_size is not None:
            width, height = self.fallback_frame_size
            if width <= 0 or height <= 0:
                raise ValueError("fallback frame dimensions must be positive")


@dataclass(frozen=True)
class FrameContext:
    frame_index: int
    frame_ref: str
    width: int
    height: int
    size_source: str
    has_real_frame_reference: bool


@dataclass(frozen=True)
class OutletEvidence:
    observation: Observation
    distance_ratio: float
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class AdapterResult:
    video_id: str
    observations: tuple[SpatialObservation, ...]
    metrics: Mapping[str, Any]
    hashes: Mapping[str, Any]
    output_paths: Mapping[str, str]


T = TypeVar("T", bound=BaseModel)


def _load_jsonl(path: Path, model: type[T]) -> list[T]:
    records: list[T] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            try:
                records.append(model.model_validate(value))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def _contains_forbidden_truth_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"region_id", "anchor_id"} or _contains_forbidden_truth_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_truth_key(item) for item in value)
    return False


def load_space_vocabulary(path: str | Path) -> Vocabulary:
    """Load an inference-only vocabulary with exactly the six allowed concepts."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if _contains_forbidden_truth_key(raw):
        raise ValueError("space inference vocabulary contains forbidden truth IDs")
    vocabulary = load_vocabulary(source)
    actual = {entry.canonical_id for entry in vocabulary.entries}
    if actual != ALLOWED_SPACE_CONCEPTS:
        missing = sorted(ALLOWED_SPACE_CONCEPTS - actual)
        unexpected = sorted(actual - ALLOWED_SPACE_CONCEPTS)
        raise ValueError(
            f"space vocab concept set mismatch; missing={missing}, unexpected={unexpected}"
        )
    return vocabulary


_FRAME_INDEX_RE = re.compile(r"(?:^|_)f(?P<index>\d+)(?:_|$)")
_KEYFRAME_RE = re.compile(r"^kf_(?P<index>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _frame_index(observation_id: str) -> int:
    match = _FRAME_INDEX_RE.search(observation_id)
    if not match:
        raise ValueError(
            f"observation_id {observation_id!r} does not contain a frame index"
        )
    return int(match.group("index"))


def _keyframes_by_index(ingest_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    keyframes_dir = ingest_dir / "keyframes"
    if not keyframes_dir.exists():
        return result
    for path in sorted(keyframes_dir.iterdir()):
        if not path.is_file():
            continue
        match = _KEYFRAME_RE.match(path.name)
        if not match:
            continue
        index = int(match.group("index"))
        if index in result:
            raise ValueError(f"duplicate keyframe index {index} in {keyframes_dir}")
        result[index] = path
    return result


class _FrameResolver:
    def __init__(
        self,
        *,
        video_id: str,
        ingest_dir: Path,
        fallback_frame_size: tuple[int, int] | None,
    ) -> None:
        self.video_id = video_id
        self.paths = _keyframes_by_index(ingest_dir)
        self.fallback_frame_size = fallback_frame_size
        self.cache: dict[int, FrameContext] = {}

    def get(self, frame_index: int) -> FrameContext:
        cached = self.cache.get(frame_index)
        if cached is not None:
            return cached
        path = self.paths.get(frame_index)
        if path is not None:
            with Image.open(path) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid keyframe dimensions: {path}")
            context = FrameContext(
                frame_index=frame_index,
                frame_ref=path.as_posix(),
                width=width,
                height=height,
                size_source="pil",
                has_real_frame_reference=True,
            )
        elif self.fallback_frame_size is not None:
            width, height = self.fallback_frame_size
            context = FrameContext(
                frame_index=frame_index,
                frame_ref=f"{self.video_id}:frame:{frame_index:06d}",
                width=width,
                height=height,
                size_source="explicit_fallback",
                has_real_frame_reference=False,
            )
        else:
            raise FileNotFoundError(
                f"keyframe kf_{frame_index:06d} missing and no explicit frame size supplied"
            )
        self.cache[frame_index] = context
        return context


def _canonical_for_tracklet(tracklet: Tracklet, vocabulary: Vocabulary) -> str | None:
    raw_label = tracklet.attributes.get("label", "")
    if not raw_label:
        return None
    return vocabulary.match(raw_label).canonical_id


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _outlet_evidence_refs(
    observation: Observation, frame: FrameContext, distance_ratio: float
) -> tuple[str, ...]:
    refs: list[str] = []
    if observation.crop_ref.strip():
        refs.append(f"outlet_crop:{observation.crop_ref.strip()}")
    if frame.has_real_frame_reference:
        refs.append(
            f"outlet_frame:{frame.frame_ref}#observation={observation.observation_id}"
        )
    if refs:
        refs.append(f"outlet_center_distance_ratio:{distance_ratio:.8f}")
    return tuple(sorted(refs))


def _nearest_outlet(
    furniture: Observation,
    frame: FrameContext,
    outlets: Iterable[Observation],
    threshold: float,
) -> OutletEvidence | None:
    furniture_center = _bbox_center(furniture.bbox)
    diagonal = math.hypot(frame.width, frame.height)
    eligible: list[OutletEvidence] = []
    for outlet in outlets:
        outlet_center = _bbox_center(outlet.bbox)
        distance_ratio = math.dist(furniture_center, outlet_center) / diagonal
        if distance_ratio >= threshold:
            continue
        refs = _outlet_evidence_refs(outlet, frame, distance_ratio)
        if not refs:
            continue
        eligible.append(
            OutletEvidence(
                observation=outlet,
                distance_ratio=distance_ratio,
                evidence_refs=refs,
            )
        )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (item.distance_ratio, item.observation.observation_id),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    separators = None if indent is not None else (",", ":")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=separators,
        )
        + "\n"
    ).encode("utf-8")


def _normalized_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.write_bytes(value)


def adapt_space_observations(
    *,
    ingest_dir: str | Path,
    vocab_path: str | Path,
    out_dir: str | Path,
    video_id: str | None = None,
    config: AdapterConfig | None = None,
) -> AdapterResult:
    """Convert ingest tracklets into deterministic automatic observations."""

    ingest_dir = Path(ingest_dir)
    out_dir = Path(out_dir)
    config = config or AdapterConfig()
    observations_path = ingest_dir / "observations.jsonl"
    tracklets_path = ingest_dir / "tracklets.jsonl"
    vocabulary_path = Path(vocab_path)
    observations = _load_jsonl(observations_path, Observation)
    tracklets = _load_jsonl(tracklets_path, Tracklet)
    vocabulary = load_space_vocabulary(vocabulary_path)

    all_video_ids = {
        *(observation.video_id for observation in observations),
        *(tracklet.video_id for tracklet in tracklets),
    }
    if not all_video_ids:
        if not video_id:
            raise ValueError("cannot infer video_id from empty ingest artifacts")
        selected_video_id = video_id
    elif len(all_video_ids) != 1:
        raise ValueError(f"ingest directory mixes video IDs: {sorted(all_video_ids)}")
    else:
        selected_video_id = next(iter(all_video_ids))
    if video_id is not None and video_id != selected_video_id:
        raise ValueError(
            f"requested video_id {video_id!r} does not match ingest {selected_video_id!r}"
        )

    observation_by_id = {observation.observation_id: observation for observation in observations}
    if len(observation_by_id) != len(observations):
        raise ValueError("duplicate observation_id in observations.jsonl")

    concept_by_tracklet: dict[str, str] = {}
    observation_owner: dict[str, str] = {}
    for tracklet in tracklets:
        canonical = _canonical_for_tracklet(tracklet, vocabulary)
        if canonical is not None:
            concept_by_tracklet[tracklet.tracklet_id] = canonical
        for observation_id in tracklet.observation_ids:
            if observation_id not in observation_by_id:
                raise ValueError(
                    f"tracklet {tracklet.tracklet_id} references missing observation "
                    f"{observation_id}"
                )
            previous = observation_owner.setdefault(observation_id, tracklet.tracklet_id)
            if previous != tracklet.tracklet_id:
                raise ValueError(
                    f"observation {observation_id} belongs to multiple tracklets"
                )

    resolver = _FrameResolver(
        video_id=selected_video_id,
        ingest_dir=ingest_dir,
        fallback_frame_size=config.fallback_frame_size,
    )
    outlets_by_frame: dict[int, list[Observation]] = defaultdict(list)
    for tracklet in tracklets:
        if concept_by_tracklet.get(tracklet.tracklet_id) != OUTLET_CONCEPT:
            continue
        for observation_id in tracklet.observation_ids:
            observation = observation_by_id[observation_id]
            outlets_by_frame[_frame_index(observation.observation_id)].append(observation)
    for items in outlets_by_frame.values():
        items.sort(key=lambda observation: observation.observation_id)

    output: list[SpatialObservation] = []
    concept_tracklet_counts: Counter[str] = Counter()
    ignored_tracklets = 0
    for tracklet in sorted(tracklets, key=lambda item: item.tracklet_id):
        canonical = concept_by_tracklet.get(tracklet.tracklet_id)
        if canonical == OUTLET_CONCEPT:
            continue
        spec = REGION_SPECS.get(canonical or "")
        if spec is None:
            ignored_tracklets += 1
            continue
        concept_tracklet_counts[canonical] += 1
        track_observations = sorted(
            (observation_by_id[item] for item in tracklet.observation_ids),
            key=lambda observation: (
                _frame_index(observation.observation_id),
                observation.timestamp_ms,
                observation.observation_id,
            ),
        )
        for observation in track_observations:
            frame_index = _frame_index(observation.observation_id)
            frame = resolver.get(frame_index)
            outlet = _nearest_outlet(
                observation,
                frame,
                outlets_by_frame.get(frame_index, []),
                config.near_diagonal_ratio,
            )
            furniture_evidence = [
                f"frame:{frame.frame_ref}#observation={observation.observation_id}"
            ]
            if observation.crop_ref.strip():
                furniture_evidence.append(f"crop:{observation.crop_ref.strip()}")
            output.append(
                SpatialObservation(
                    video_id=selected_video_id,
                    timestamp_ms=observation.timestamp_ms,
                    frame_ref=frame.frame_ref,
                    bbox=observation.bbox,
                    region_track_id=tracklet.tracklet_id,
                    anchor_label=spec.anchor_label,
                    display_name_zh=spec.display_name_zh,
                    support_type=spec.support_type,
                    capacity_class=spec.capacity_class,
                    power_state=(PowerState.NEAR if outlet else PowerState.UNKNOWN),
                    power_evidence_refs=(list(outlet.evidence_refs) if outlet else []),
                    evidence_refs=sorted(furniture_evidence),
                    model_confidence=observation.quality,
                    anchor_confidence=observation.quality,
                    support_confidence=1.0,
                    capacity_confidence=1.0,
                    power_confidence=(outlet.observation.quality if outlet else None),
                    model_version=observation.model_version,
                )
            )

    output.sort(
        key=lambda item: (
            item.region_track_id or "",
            item.timestamp_ms,
            item.frame_ref,
            item.bbox or (),
        )
    )
    output_bytes = b"".join(
        _canonical_json_bytes(item.model_dump(mode="json")).rstrip(b"\n") + b"\n"
        for item in output
    )
    frame_sources = Counter(context.size_source for context in resolver.cache.values())
    power_counts = Counter(item.power_state.value for item in output)
    detected_concepts = sorted(
        concept for concept, count in concept_tracklet_counts.items() if count > 0
    )
    metrics: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "video_id": selected_video_id,
        "near_diagonal_ratio": config.near_diagonal_ratio,
        "fallback_frame_size": (
            list(config.fallback_frame_size) if config.fallback_frame_size else None
        ),
        "input_observation_count": len(observations),
        "input_tracklet_count": len(tracklets),
        "furniture_tracklet_count": sum(concept_tracklet_counts.values()),
        "outlet_tracklet_count": sum(
            concept == OUTLET_CONCEPT for concept in concept_by_tracklet.values()
        ),
        "ignored_tracklet_count": ignored_tracklets,
        "auto_observation_count": len(output),
        "target_concept_count": len(REGION_SPECS),
        "detected_target_concept_count": len(detected_concepts),
        "detected_target_concepts": detected_concepts,
        "concept_tracklet_counts": dict(sorted(concept_tracklet_counts.items())),
        "power_state_counts": {
            state.value: power_counts.get(state.value, 0) for state in PowerState
        },
        "frame_size_source_counts": dict(sorted(frame_sources.items())),
    }
    metrics_bytes = _canonical_json_bytes(metrics, indent=2)

    frame_dimension_payload = {
        str(index): {
            "frame_ref": context.frame_ref,
            "width": context.width,
            "height": context.height,
            "source": context.size_source,
        }
        for index, context in sorted(resolver.cache.items())
    }
    input_hashes = {
        "observations_jsonl": _sha256_file(observations_path),
        "tracklets_jsonl": _sha256_file(tracklets_path),
        "space_vocab_json": _sha256_file(vocabulary_path),
        "frame_dimensions": _normalized_hash(frame_dimension_payload),
    }
    output_hashes = {
        AUTO_OBSERVATIONS_FILENAME: hashlib.sha256(output_bytes).hexdigest(),
        METRICS_FILENAME: hashlib.sha256(metrics_bytes).hexdigest(),
    }
    hashes: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "algorithm": "sha256",
        "inputs": input_hashes,
        "outputs": output_hashes,
        "normalized_hash": _normalized_hash(
            {
                "adapter_version": ADAPTER_VERSION,
                "inputs": input_hashes,
                "outputs": output_hashes,
            }
        ),
    }
    hashes_bytes = _canonical_json_bytes(hashes, indent=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    auto_path = out_dir / AUTO_OBSERVATIONS_FILENAME
    metrics_path = out_dir / METRICS_FILENAME
    hashes_path = out_dir / HASHES_FILENAME
    _write_bytes(auto_path, output_bytes)
    _write_bytes(metrics_path, metrics_bytes)
    _write_bytes(hashes_path, hashes_bytes)
    return AdapterResult(
        video_id=selected_video_id,
        observations=tuple(output),
        metrics=metrics,
        hashes=hashes,
        output_paths={
            "auto_observations": str(auto_path),
            "metrics": str(metrics_path),
            "hashes": str(hashes_path),
        },
    )


def _parse_frame_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("frame size must be WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("frame dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest-dir", required=True, type=Path)
    parser.add_argument(
        "--vocab",
        type=Path,
        default=PROJ / "fixtures" / "hero_s1" / "space_vocab.json",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--near-diagonal-ratio", type=float, default=0.15)
    parser.add_argument(
        "--frame-size",
        type=_parse_frame_size,
        default=None,
        help="explicit WIDTHxHEIGHT fallback for tests when keyframe images are absent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = adapt_space_observations(
        ingest_dir=args.ingest_dir,
        vocab_path=args.vocab,
        out_dir=args.out_dir,
        video_id=args.video_id,
        config=AdapterConfig(
            near_diagonal_ratio=args.near_diagonal_ratio,
            fallback_frame_size=args.frame_size,
        ),
    )
    print(
        json.dumps(
            {
                "video_id": result.video_id,
                "auto_observations": result.metrics["auto_observation_count"],
                "detected_target_concepts": result.metrics[
                    "detected_target_concept_count"
                ],
                "normalized_hash": result.hashes["normalized_hash"],
                "outputs": result.output_paths,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
