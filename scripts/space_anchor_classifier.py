#!/usr/bin/env python
"""Build automatic spatial-anchor hypotheses with the Spark-local Nemotron VLM.

The command consumes only automatic ingest artifacts and the configured anchor
vocabulary.  It never accepts a hand-authored region manifest, visual review,
candidate override, or manual track mapping.  For every automatic furniture
track it creates a contact sheet (scene + highlighted target + crop views), asks
the local VLM for calibrated anchor/support/capacity hypotheses, and emits the
strict assignment contract consumed by ``space_task.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import Tracklet  # noqa: E402
from backend.tools.spatial import (  # noqa: E402
    AutomaticAnchorCandidate,
    PowerState,
    SpatialObservation,
    load_observations_jsonl,
)

CLASSIFIER_SCHEMA_VERSION = "1.0"
CLASSIFIER_VERSION = "space-anchor-nemotron-v1"
ANCHOR_CANDIDATES_FILENAME = "anchor_candidates.json"
METRICS_FILENAME = "metrics.json"
HASHES_FILENAME = "hashes.json"
DEFAULT_MODEL = (
    "/models/nv-community__NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"
)

_FRAME_RE = re.compile(r"(?:^|_)f(?P<index>\d+)(?:_|\.|$)")
_print_lock = threading.Lock()


@dataclass(frozen=True)
class TrackEvidence:
    track_id: str
    observations: tuple[SpatialObservation, ...]
    prototype_refs: tuple[str, ...]
    hero_ref: str | None
    visual_instance_id: str


def _canonical_anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold().strip()).strip("_")


def _expected_anchors(values: Sequence[str]) -> list[str]:
    result = [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]
    canonical = [_canonical_anchor(item) for item in result]
    if not result:
        raise ValueError("at least one --expected-anchor is required")
    if any(not item for item in canonical):
        raise ValueError("expected anchors must contain letters or numbers")
    if len(canonical) != len(set(canonical)):
        raise ValueError("expected anchors contain duplicates")
    return [value for _, value in sorted(zip(canonical, result, strict=True))]


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJ / path


def _load_tracklets(path: Path) -> dict[str, Tracklet]:
    result: dict[str, Tracklet] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                tracklet = Tracklet.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if tracklet.tracklet_id in result:
                raise ValueError(f"duplicate tracklet_id: {tracklet.tracklet_id}")
            result[tracklet.tracklet_id] = tracklet
    return result


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self.parent[second] = first


def automatic_visual_instance_ids(
    grouped: Mapping[str, Sequence[SpatialObservation]],
    *,
    min_shared_frames: int = 2,
    min_median_iou: float = 0.80,
) -> dict[str, str]:
    """Group parallel category tracks that cover the same physical object."""

    if min_shared_frames < 1:
        raise ValueError("min_shared_frames must be positive")
    if not 0.0 <= min_median_iou <= 1.0:
        raise ValueError("min_median_iou must be in [0, 1]")
    track_ids = sorted(grouped)
    union = _UnionFind(track_ids)
    frame_boxes: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    tracks_by_frame: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for observation in grouped[track_id]:
            if observation.bbox is None:
                continue
            boxes[observation.frame_ref] = observation.bbox
        frame_boxes[track_id] = boxes
        for frame_ref in boxes:
            tracks_by_frame[frame_ref].append(track_id)

    overlaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for frame_ref in sorted(tracks_by_frame):
        frame_tracks = sorted(set(tracks_by_frame[frame_ref]))
        for left_index, left in enumerate(frame_tracks):
            for right in frame_tracks[left_index + 1 :]:
                overlaps[(left, right)].append(
                    _bbox_iou(frame_boxes[left][frame_ref], frame_boxes[right][frame_ref])
                )
    for (left, right), values in sorted(overlaps.items()):
        eligible = sorted(value for value in values if value >= min_median_iou)
        if len(eligible) < min_shared_frames:
            continue
        median = eligible[len(eligible) // 2]
        if median >= min_median_iou:
            union.union(left, right)

    members: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        members[union.find(track_id)].append(track_id)
    instance_by_track: dict[str, str] = {}
    for items in sorted((sorted(value) for value in members.values()), key=lambda x: x[0]):
        digest = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]
        instance_id = f"auto_visual_{digest}"
        for track_id in items:
            instance_by_track[track_id] = instance_id
    return instance_by_track


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    source = image.convert("RGB").copy()
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (118, 118, 118))
    canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return canvas


def _expanded_crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    padding_ratio: float = 0.12,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    bounds = (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(image.width, math.ceil(x2 + pad_x)),
        min(image.height, math.ceil(y2 + pad_y)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("bbox does not intersect the evidence frame")
    return image.crop(bounds)


def _frame_index(value: str) -> int | None:
    match = _FRAME_RE.search(Path(value).name)
    return int(match.group("index")) if match else None


def _representative_observation(
    observations: Sequence[SpatialObservation], hero_ref: str | None
) -> SpatialObservation:
    hero_index = _frame_index(hero_ref or "")
    if hero_index is not None:
        matching = [
            observation
            for observation in observations
            if _frame_index(observation.frame_ref) == hero_index
        ]
        if matching:
            return max(
                matching,
                key=lambda item: (item.model_confidence, item.timestamp_ms, item.frame_ref),
            )
    return max(
        observations,
        key=lambda item: (
            item.model_confidence,
            ((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            if item.bbox is not None
            else 0.0,
            -item.timestamp_ms,
        ),
    )


def build_contact_sheet(evidence: TrackEvidence) -> tuple[bytes, list[str]]:
    """Render scene context and up to three automatic crop views into one image."""

    representative = _representative_observation(evidence.observations, evidence.hero_ref)
    if representative.bbox is None:
        raise ValueError(f"{evidence.track_id}: representative bbox missing")
    frame_path = _resolve_path(representative.frame_ref)
    if not frame_path.exists():
        raise FileNotFoundError(f"{evidence.track_id}: frame missing: {frame_path}")

    with Image.open(frame_path) as opened:
        frame = opened.convert("RGB")
    scene = frame.copy()
    draw = ImageDraw.Draw(scene)
    draw.rectangle(representative.bbox, outline=(255, 0, 0), width=max(4, scene.width // 240))
    panels: list[Image.Image] = [scene, _expanded_crop(frame, representative.bbox)]
    sources = [str(frame_path)]
    for ref in evidence.prototype_refs:
        path = _resolve_path(ref)
        if not path.exists() or path == frame_path:
            continue
        with Image.open(path) as opened:
            panels.append(opened.convert("RGB"))
        sources.append(str(path))
        if len(panels) == 4:
            break
    while len(panels) < 4:
        panels.append(panels[-1].copy())

    canvas = Image.new("RGB", (1024, 1024), (92, 92, 92))
    labels = ("SCENE - RED TARGET", "TARGET ZOOM", "SECOND VIEW", "THIRD VIEW")
    for index, panel in enumerate(panels[:4]):
        tile = _letterbox(panel, (512, 512))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle((0, 0, 511, 26), fill=(0, 0, 0))
        tile_draw.text((8, 7), labels[index], fill=(255, 255, 255))
        canvas.paste(tile, ((index % 2) * 512, (index // 2) * 512))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), sources


def _json_schema(anchors: Sequence[str]) -> dict[str, Any]:
    score_properties = {
        anchor: {"type": "integer", "minimum": 0, "maximum": 100}
        for anchor in anchors
    }
    score_properties["other"] = {"type": "integer", "minimum": 0, "maximum": 100}
    return {
        "type": "object",
        "properties": {
            "anchor_scores": {
                "type": "object",
                "properties": score_properties,
                "required": [*anchors, "other"],
                "additionalProperties": False,
            },
            "best_anchor": {"type": "string", "enum": [*anchors, "other"]},
            "display_name_zh": {"type": "string", "maxLength": 24},
            "support_type": {
                "type": "string",
                "enum": ["surface", "shelf", "floor", "unknown"],
            },
            "support_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "capacity_class": {
                "type": "string",
                "enum": ["small", "medium", "large", "unknown"],
            },
            "capacity_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "anchor_scores",
            "best_anchor",
            "display_name_zh",
            "support_type",
            "support_confidence",
            "capacity_class",
            "capacity_confidence",
        ],
        "additionalProperties": False,
    }


def _prompt(anchors: Sequence[str]) -> str:
    categories = ", ".join(f'"{anchor}" ({anchor.replace("_", " ")})' for anchor in anchors)
    return (
        "The image is an automatically generated evidence sheet from a new-home video. "
        "Top-left is the room scene; the RED rectangle is the target object/usable region. "
        "The other panels are automatic zoomed views of the same tracked object. Ignore "
        "other furniture outside the red rectangle. Classify the target independently; "
        "do not infer from crop framing or assume that it belongs to a requested class. "
        f"Allowed anchor classes are: {categories}; use other when none fits. "
        "Return integer anchor_scores from 0 to 100 as calibrated confidence for every "
        "class (scores need not sum to 100) and best_anchor. support_type means the usable "
        "placement relation: surface for a tabletop/top, shelf for a shelf/compartment, "
        "floor only for a floor zone, unknown if not visible. capacity_class is visual "
        "relative usable capacity, not exact measurement: small = one narrow shelf or one "
        "cabinet compartment; medium = a normal tabletop/dresser top holding several items; "
        "large = a room-scale broad surface or multiple full shelves. Give independent "
        "support/capacity confidence and a short Chinese display name. Output JSON only."
    )


class Client:
    def __init__(self, endpoint: str, model: str, anchors: Sequence[str], guided: bool) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.anchors = list(anchors)
        self.guided = guided
        self.schema = _json_schema(self.anchors)
        self.prompt = _prompt(self.anchors)
        self.usage = {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.lock = threading.Lock()

    def chat(self, image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        if self.guided:
            payload["guided_json"] = self.schema
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        usage = result.get("usage", {})
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        return str(result["choices"][0]["message"]["content"])


def parse_prediction(text: str, anchors: Sequence[str]) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    expected_scores = {*anchors, "other"}
    scores = payload.get("anchor_scores")
    if not isinstance(scores, dict) or set(scores) != expected_scores:
        return None
    try:
        normalized_scores = {
            anchor: max(0, min(100, int(scores[anchor]))) for anchor in sorted(scores)
        }
        best_anchor = str(payload["best_anchor"])
        support_type = str(payload["support_type"])
        capacity_class = str(payload["capacity_class"])
        support_confidence = max(0, min(100, int(payload["support_confidence"])))
        capacity_confidence = max(0, min(100, int(payload["capacity_confidence"])))
    except (KeyError, TypeError, ValueError):
        return None
    if best_anchor not in expected_scores:
        return None
    if support_type not in {"surface", "shelf", "floor", "unknown"}:
        return None
    if capacity_class not in {"small", "medium", "large", "unknown"}:
        return None
    return {
        "anchor_scores": normalized_scores,
        "best_anchor": best_anchor,
        "display_name_zh": str(payload.get("display_name_zh", "自动识别区域")).strip()[:24]
        or "自动识别区域",
        "support_type": support_type,
        "support_confidence": support_confidence,
        "capacity_class": capacity_class,
        "capacity_confidence": capacity_confidence,
    }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[str(record["key"])] = dict(record["prediction"])
    return cache


def _aggregate_power(
    observations: Sequence[SpatialObservation], min_confidence: float = 0.70
) -> tuple[PowerState, float | None, list[str]]:
    valid: list[tuple[float, list[str]]] = []
    for observation in observations:
        confidence = observation.power_confidence
        refs = sorted({ref.strip() for ref in observation.power_evidence_refs if ref.strip()})
        if (
            observation.power_state is PowerState.NEAR
            and confidence is not None
            and confidence >= min_confidence
            and refs
        ):
            valid.append((confidence, refs))
    if not valid:
        return PowerState.UNKNOWN, None, []
    return (
        PowerState.NEAR,
        round(max(item[0] for item in valid), 6),
        sorted({ref for _, refs in valid for ref in refs}),
    )


def _candidate_from_prediction(
    evidence: TrackEvidence,
    prediction: Mapping[str, Any],
    *,
    contact_ref: str,
    contact_sha256: str,
    model: str,
) -> AutomaticAnchorCandidate:
    observation_count = len(evidence.observations)
    support = None if prediction["support_type"] == "unknown" else prediction["support_type"]
    capacity = (
        None if prediction["capacity_class"] == "unknown" else prediction["capacity_class"]
    )
    power_state, power_confidence, power_refs = _aggregate_power(evidence.observations)
    hypotheses = []
    for anchor in sorted(prediction["anchor_scores"]):
        if anchor == "other":
            continue
        score = float(prediction["anchor_scores"][anchor]) / 100.0
        vote_count = max(1, min(observation_count, int(round(score * observation_count))))
        hypotheses.append(
            {
                "anchor": anchor,
                "label_vote_count": vote_count,
                "mean_confidence": score,
                "max_confidence": score,
                "proposal_display_name_zh": prediction["display_name_zh"],
                "support_type": support,
                "support_confidence": float(prediction["support_confidence"]) / 100.0,
                "capacity_class": capacity,
                "capacity_confidence": float(prediction["capacity_confidence"]) / 100.0,
            }
        )
    detector_versions = {
        item.model_version.strip() for item in evidence.observations if item.model_version.strip()
    }
    original_refs = {
        ref.strip()
        for item in evidence.observations
        for ref in item.evidence_refs
        if ref.strip()
    }
    return AutomaticAnchorCandidate.model_validate(
        {
            "candidate_id": f"auto_anchor_{evidence.track_id}",
            "visual_instance_id": evidence.visual_instance_id,
            "observation_count": observation_count,
            "display_name_zh": prediction["display_name_zh"],
            "power_state": power_state.value,
            "power_confidence": power_confidence,
            "power_evidence_refs": power_refs,
            "evidence_refs": sorted(
                {
                    *original_refs,
                    f"auto_vlm_contact:{contact_ref}@sha256:{contact_sha256}",
                }
            ),
            "source_track_ids": [evidence.track_id],
            "model_versions": sorted({*detector_versions, f"{CLASSIFIER_VERSION}@{model}"}),
            "anchor_hypotheses": hypotheses,
        }
    )


def classify_one(
    client: Client,
    evidence: TrackEvidence,
    *,
    out_dir: Path,
    evidence_dir: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
) -> tuple[AutomaticAnchorCandidate | None, dict[str, Any]]:
    image_bytes, sources = build_contact_sheet(evidence)
    image_sha = _sha256_bytes(image_bytes)
    schema_sha = _sha256_bytes(_json_bytes(client.schema))
    cache_key = f"{image_sha}:{schema_sha}:{CLASSIFIER_VERSION}:{client.model}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    contact_path = evidence_dir / f"{evidence.track_id}.jpg"
    contact_path.write_bytes(image_bytes)
    with cache_lock:
        prediction = cache.get(cache_key)
    cache_hit = prediction is not None
    if prediction is None:
        for attempt in range(2):
            try:
                prediction = parse_prediction(client.chat(image_bytes), client.anchors)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                with client.lock:
                    client.usage["errors"] += 1
                with _print_lock:
                    print(f"[warn] {evidence.track_id} attempt {attempt + 1}: {exc}", file=sys.stderr)
                prediction = None
            if prediction is not None:
                break
        if prediction is not None:
            with cache_lock:
                cache[cache_key] = prediction
                with cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"key": cache_key, "prediction": prediction},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    diagnostic = {
        "track_id": evidence.track_id,
        "visual_instance_id": evidence.visual_instance_id,
        "contact_sha256": image_sha,
        "contact_ref": (
            contact_path.relative_to(PROJ).as_posix()
            if contact_path.is_relative_to(PROJ)
            else contact_path.as_posix()
        ),
        "source_refs": sources,
        "cache_hit": cache_hit,
        "status": "OK" if prediction is not None else "CLASSIFICATION_FAILED",
        "prediction": prediction,
    }
    if prediction is None:
        return None, diagnostic
    return (
        _candidate_from_prediction(
            evidence,
            prediction,
            contact_ref=diagnostic["contact_ref"],
            contact_sha256=image_sha,
            model=client.model,
        ),
        diagnostic,
    )


def run_classifier(
    *,
    observations_path: Path,
    ingest_dir: Path,
    out_dir: Path,
    evidence_dir: Path | None = None,
    expected_anchors: Sequence[str],
    endpoint: str,
    model: str,
    concurrency: int = 8,
    guided: bool = True,
    max_failed_rate: float = 0.02,
    limit: int = 0,
) -> tuple[list[AutomaticAnchorCandidate], dict[str, Any], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    observations = load_observations_jsonl(observations_path)
    grouped: dict[str, list[SpatialObservation]] = defaultdict(list)
    for observation in observations:
        if not observation.region_track_id:
            raise ValueError("all classifier observations must have automatic region_track_id")
        grouped[observation.region_track_id].append(observation)
    tracklets = _load_tracklets(ingest_dir / "tracklets.jsonl")
    missing = sorted(set(grouped) - set(tracklets))
    if missing:
        raise ValueError(f"automatic tracks missing from ingest: {missing[:5]}")
    instance_ids = automatic_visual_instance_ids(grouped)
    evidence_rows: list[TrackEvidence] = []
    for track_id in sorted(grouped):
        tracklet = tracklets[track_id]
        evidence_rows.append(
            TrackEvidence(
                track_id=track_id,
                observations=tuple(
                    sorted(grouped[track_id], key=lambda item: (item.timestamp_ms, item.frame_ref))
                ),
                prototype_refs=tuple(tracklet.prototype_refs),
                hero_ref=str(tracklet.attributes.get("hero_ref") or "").strip() or None,
                visual_instance_id=instance_ids[track_id],
            )
        )
    if limit:
        evidence_rows = evidence_rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = evidence_dir or out_dir / "evidence"
    cache_path = out_dir / "anchor_predictions.cache.jsonl"
    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()
    client = Client(endpoint, model, expected_anchors, guided)
    candidates: list[AutomaticAnchorCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                classify_one,
                client,
                evidence,
                out_dir=out_dir,
                evidence_dir=evidence_dir,
                cache=cache,
                cache_path=cache_path,
                cache_lock=cache_lock,
            ): evidence.track_id
            for evidence in evidence_rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            track_id = futures[future]
            try:
                candidate, diagnostic = future.result()
            except Exception as exc:  # keep full automatic-batch evidence, fail at the gate below
                candidate = None
                diagnostic = {
                    "track_id": track_id,
                    "status": "CLASSIFICATION_EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            diagnostics.append(diagnostic)
            if candidate is not None:
                candidates.append(candidate)
            if completed % 10 == 0 or completed == len(futures):
                with _print_lock:
                    print(f"[space-anchor] {completed}/{len(futures)}", flush=True)

    candidates.sort(key=lambda item: item.candidate_id)
    diagnostics.sort(key=lambda item: item["track_id"])
    failed = len(evidence_rows) - len(candidates)
    failed_rate = failed / len(evidence_rows) if evidence_rows else 1.0
    if failed_rate > max_failed_rate:
        raise RuntimeError(
            f"automatic anchor classification failure rate {failed_rate:.4f} "
            f"exceeds {max_failed_rate:.4f}"
        )

    candidate_payload = [item.model_dump(mode="json") for item in candidates]
    candidates_bytes = _json_bytes(candidate_payload, indent=2)
    metrics: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "expected_anchor_labels": list(expected_anchors),
        "input_observation_count": len(observations),
        "input_track_count": len(grouped),
        "classified_track_count": len(candidates),
        "failed_track_count": failed,
        "failed_rate": round(failed_rate, 8),
        "automatic_visual_instance_count": len(set(instance_ids.values())),
        "model": model,
        "endpoint": endpoint,
        "guided_json": guided,
        "concurrency": concurrency,
        "usage": dict(client.usage),
        "diagnostics": diagnostics,
    }
    metrics_bytes = _json_bytes(metrics, indent=2)
    hashes: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "algorithm": "sha256",
        "inputs": {
            "auto_observations.jsonl": _sha256_file(observations_path),
            "tracklets.jsonl": _sha256_file(ingest_dir / "tracklets.jsonl"),
            "expected_anchors": _sha256_bytes(_json_bytes(list(expected_anchors)).rstrip(b"\n")),
        },
        "outputs": {
            ANCHOR_CANDIDATES_FILENAME: _sha256_bytes(candidates_bytes),
            METRICS_FILENAME: _sha256_bytes(metrics_bytes),
        },
    }
    hashes["normalized_hash"] = _sha256_bytes(_json_bytes(hashes).rstrip(b"\n"))
    (out_dir / ANCHOR_CANDIDATES_FILENAME).write_bytes(candidates_bytes)
    (out_dir / METRICS_FILENAME).write_bytes(metrics_bytes)
    (out_dir / HASHES_FILENAME).write_bytes(_json_bytes(hashes, indent=2))
    return candidates, metrics, hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--ingest-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="contact sheets; set to local-data so pull_results only transfers JSON",
    )
    parser.add_argument("--expected-anchor", action="append", default=[])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-failed-rate", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-guided", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    anchors = _expected_anchors(args.expected_anchor)
    candidates, metrics, hashes = run_classifier(
        observations_path=args.observations,
        ingest_dir=args.ingest_dir,
        out_dir=args.out_dir,
        evidence_dir=args.evidence_dir,
        expected_anchors=anchors,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        guided=not args.no_guided,
        max_failed_rate=args.max_failed_rate,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "classified_tracks": len(candidates),
                "visual_instances": metrics["automatic_visual_instance_count"],
                "failed_tracks": metrics["failed_track_count"],
                "normalized_hash": hashes["normalized_hash"],
                "output": str(args.out_dir / ANCHOR_CANDIDATES_FILENAME),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
