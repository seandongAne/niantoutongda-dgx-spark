"""Ground-truth-free v5/v6 ingest diagnostics.

This module deliberately reports artifact counts, canonical coverage and wall
clock only.  It cannot emit the frozen hardval recall/fragmentation/FP metrics;
those require manually boxed task-A ground truth and live in detection_eval.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.pipeline.vocab import Vocabulary


_LOG_LINE = re.compile(
    r"^\[(?P<video>[^]]+)] (?P<keyframes>\d+) kf, (?P<observations>\d+) obs, "
    r"(?P<tracklets>\d+) tracklets, (?P<wall>\d+)s"
    r"(?: \(detect=(?P<detect>[0-9.]+)s, batch=(?P<batch>\d+), tiled_kf=(?P<tiled>\d+)"
    r"(?:, tile_mode=(?P<tile_mode>[a-z_]+))?\))?$"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def _artifact_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def summarize_ingest(root: str | Path, vocab: Vocabulary) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest root not found: {root}")
    videos: dict[str, dict[str, Any]] = {}
    artifact_paths: list[Path] = []
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        tracklet_path = video_dir / "tracklets.jsonl"
        observation_path = video_dir / "observations.jsonl"
        if not tracklet_path.exists() or not observation_path.exists():
            continue
        tracklets = _jsonl(tracklet_path)
        observations = _jsonl(observation_path)
        artifact_paths.extend((observation_path, tracklet_path))
        canonical_counts: Counter[str] = Counter()
        raw_unknown_counts: Counter[str] = Counter()
        hero_count = 0
        for tracklet in tracklets:
            attributes = tracklet.get("attributes") or {}
            raw_label = str(attributes.get("label", ""))
            match = vocab.match(raw_label)
            if match.canonical_id is None:
                canonical_counts["__unknown__"] += 1
                raw_unknown_counts[raw_label or "__empty__"] += 1
            else:
                canonical_counts[match.canonical_id] += 1
            if attributes.get("hero_ref"):
                hero_count += 1
        videos[video_dir.name] = {
            "keyframe_count": len(list((video_dir / "keyframes").glob("*.jpg"))),
            "observation_count": len(observations),
            "tracklet_count": len(tracklets),
            "hero_ref_count": hero_count,
            "hero_ref_coverage": hero_count / len(tracklets) if tracklets else 0.0,
            "canonical_tracklet_counts": dict(sorted(canonical_counts.items())),
            "unknown_raw_label_counts": dict(sorted(raw_unknown_counts.items())),
        }
    if not videos:
        raise ValueError(f"no complete video artifacts found under {root}")
    return {
        "root": str(root),
        "artifact_sha256": _artifact_hash(artifact_paths),
        "video_count": len(videos),
        "keyframe_count": sum(item["keyframe_count"] for item in videos.values()),
        "observation_count": sum(item["observation_count"] for item in videos.values()),
        "tracklet_count": sum(item["tracklet_count"] for item in videos.values()),
        "hero_ref_count": sum(item["hero_ref_count"] for item in videos.values()),
        "videos": videos,
    }


def parse_ingest_log(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = _LOG_LINE.match(line)
        if not match:
            continue
        values = match.groupdict()
        rows[values["video"]] = {
            "wall_s": int(values["wall"]),
            "detection_s": float(values["detect"]) if values["detect"] else None,
            "frame_batch_size": int(values["batch"]) if values["batch"] else 1,
            "tiled_keyframe_count": int(values["tiled"]) if values["tiled"] else 0,
            "tile_selection_mode": values["tile_mode"] or "legacy_or_none",
        }
    return {
        "videos": rows,
        "wall_s": sum(item["wall_s"] for item in rows.values()),
        "detection_s": (
            sum(item["detection_s"] for item in rows.values())
            if rows and all(item["detection_s"] is not None for item in rows.values())
            else None
        ),
    }


def compare_ingests(
    *,
    baseline_root: str | Path,
    candidate_root: str | Path,
    vocab: Vocabulary,
    baseline_log: str | Path | None = None,
    candidate_log: str | Path | None = None,
) -> dict[str, Any]:
    baseline = summarize_ingest(baseline_root, vocab)
    candidate = summarize_ingest(candidate_root, vocab)
    baseline_timing = parse_ingest_log(baseline_log)
    candidate_timing = parse_ingest_log(candidate_log)
    baseline["timing"] = baseline_timing
    candidate["timing"] = candidate_timing

    video_ids = sorted(set(baseline["videos"]) | set(candidate["videos"]))
    per_video = {}
    for video_id in video_ids:
        before = baseline["videos"].get(video_id, {})
        after = candidate["videos"].get(video_id, {})
        per_video[video_id] = {
            "observation_count_delta": after.get("observation_count", 0)
            - before.get("observation_count", 0),
            "tracklet_count_delta": after.get("tracklet_count", 0)
            - before.get("tracklet_count", 0),
            "hero_ref_coverage": after.get("hero_ref_coverage", 0.0),
        }

    wall_delta = None
    wall_ratio = None
    if baseline_timing and candidate_timing and baseline_timing["wall_s"]:
        wall_delta = candidate_timing["wall_s"] - baseline_timing["wall_s"]
        wall_ratio = candidate_timing["wall_s"] / baseline_timing["wall_s"]
    return {
        "schema_version": "1.0",
        "comparison_type": "ground_truth_free_ingest_diagnostic",
        "hardval_metrics_evaluated": False,
        "hardval_blocker": (
            "machine-readable manually boxed task-A hardval ground truth is absent; "
            "tracklet deltas are not recall, fragmentation, or false-positive metrics"
        ),
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "observation_count": candidate["observation_count"] - baseline["observation_count"],
            "tracklet_count": candidate["tracklet_count"] - baseline["tracklet_count"],
            "wall_s": wall_delta,
            "wall_ratio_candidate_over_baseline": wall_ratio,
            "per_video": per_video,
        },
    }
