#!/usr/bin/env python
"""把任一 ingest 版本在 hardval 选帧上的检测导出为 detection_eval 预测格式。

同一份 manifest 可分别对 v5/v6 ingest 跑一次,得到三指标对照行。
未进入任何 tracklet 的检测以 untracked_<obs_id> 作为 track_id 如实导出,
不做任何过滤或修饰。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.vocab import Vocabulary  # noqa: E402
from backend.schemas.core import Observation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--manifest", required=True, help="hardval_sample.py 产出的 manifest.json")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    vocab = Vocabulary.from_json(args.vocab)
    root = Path(args.ingest_root)

    wanted: dict[str, set[str]] = {}
    for row in manifest["frames"]:
        wanted.setdefault(row["video_id"], set()).add(row["frame_id"])

    frames_out: list[dict] = []
    for video_id in sorted(wanted):
        video_dir = root / video_id
        observations = [
            Observation.model_validate_json(line)
            for line in (video_dir / "observations.jsonl").read_text().splitlines()
            if line
        ]
        track_by_obs: dict[str, str] = {}
        label_by_track: dict[str, str] = {}
        tracklets_path = video_dir / "tracklets.jsonl"
        if tracklets_path.exists():
            for line in tracklets_path.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                label_by_track[row["tracklet_id"]] = str(
                    row.get("attributes", {}).get("label", "")
                )
                for obs_id in row.get("observation_ids", []):
                    track_by_obs[obs_id] = row["tracklet_id"]

        by_frame: dict[str, list[dict]] = {}
        for ob in observations:
            frame_id = next(
                part[1:] for part in ob.observation_id.split("_") if part.startswith("f") and part[1:].isdigit()
            )
            if frame_id not in wanted[video_id]:
                continue
            track_id = track_by_obs.get(ob.observation_id, f"untracked_{ob.observation_id}")
            raw_label = label_by_track.get(track_id, "")
            canonical = vocab.match(raw_label).canonical_id or raw_label or "unlabelled"
            by_frame.setdefault(frame_id, []).append(
                {
                    "track_id": track_id,
                    "canonical_id": canonical,
                    "bbox": list(ob.bbox),
                }
            )
        for frame_id in sorted(wanted[video_id]):
            frames_out.append(
                {
                    "sequence_id": video_id,
                    "frame_id": frame_id,
                    "predictions": sorted(
                        by_frame.get(frame_id, []),
                        key=lambda p: (p["track_id"], p["canonical_id"], p["bbox"]),
                    ),
                }
            )

    payload = {"dataset_id": manifest["dataset_id"], "frames": frames_out}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frames": len(frames_out),
                "predictions": sum(len(f["predictions"]) for f in frames_out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
