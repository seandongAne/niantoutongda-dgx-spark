#!/usr/bin/env python
"""S2.5-5 hardval 帧采样:从 ingest 产物确定性抽 30–50 帧供人工框真值。

选择策略(无模型、无随机):每视频配额一半给"焦点分"最高的帧
(弱类命中×3 + 检测密度 + 小框数),一半给时间轴均匀步进,保证不只
盯着模型已经看见的东西;帧间强制最小间隔。只读 ingest,产出:

  <out>/frames/<video>_kf_<frame>.jpg   帧图副本
  <out>/manifest.json                   选帧清单与理由
  <out>/gt.skeleton.json                空 instances 的真值骨架
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.vocab import Vocabulary  # noqa: E402
from backend.schemas.core import Observation  # noqa: E402

DEFAULT_FOCUS = "water_bottle,security_camera,table_lamp,luggage,storage_box,stuffed_animal,night_light"
MIN_SPACING_MS = 2000


def _read_observations(path: Path) -> list[Observation]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [Observation.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _frame_index(observation_id: str) -> int:
    # observation_id 形如 v1_f000123_d02
    for part in observation_id.split("_"):
        if part.startswith("f") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"cannot parse frame index from {observation_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-frames", type=int, default=40)
    parser.add_argument("--focus", default=DEFAULT_FOCUS, help="comma-separated canonical_id")
    parser.add_argument("--dataset-id", default="dev_a_hardval_v1")
    args = parser.parse_args()
    if "task_b" in args.dataset_id or "dev_b" in args.dataset_id:
        parser.error("task B dataset is sealed from the technical side")

    root = Path(args.ingest_root)
    vocab = Vocabulary.from_json(args.vocab)
    focus = {item.strip() for item in args.focus.split(",") if item.strip()}
    video_dirs = sorted(d for d in root.iterdir() if (d / "observations.jsonl").exists())
    if not video_dirs:
        raise SystemExit(f"no ingest videos under {root}")
    quota, remainder = divmod(args.total_frames, len(video_dirs))

    manifest: list[dict] = []
    skeleton_frames: list[dict] = []
    frames_out = Path(args.out) / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    for index, video_dir in enumerate(video_dirs):
        video_id = video_dir.name
        per_video = quota + (1 if index < remainder else 0)
        observations = _read_observations(video_dir / "observations.jsonl")
        by_frame: dict[int, list[Observation]] = defaultdict(list)
        for ob in observations:
            by_frame[_frame_index(ob.observation_id)].append(ob)
        if not by_frame:
            continue
        areas = sorted(
            (ob.bbox[2] - ob.bbox[0]) * (ob.bbox[3] - ob.bbox[1]) for ob in observations
        )
        small_cutoff = areas[len(areas) // 4] if areas else 0.0  # 面积最小四分位

        # observations 不带 label;焦点命中从该观测所属 tracklet 的 canonical 反查
        canonical_by_obs: dict[str, str] = {}
        tracklets_path = video_dir / "tracklets.jsonl"
        if tracklets_path.exists():
            for line in tracklets_path.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                label = str(row.get("attributes", {}).get("label", ""))
                canonical = vocab.match(label).canonical_id or label
                for obs_id in row.get("observation_ids", []):
                    canonical_by_obs[obs_id] = canonical

        scored = []
        for frame, obs in by_frame.items():
            focus_hits = sum(canonical_by_obs.get(ob.observation_id, "") in focus for ob in obs)
            small = sum(
                (ob.bbox[2] - ob.bbox[0]) * (ob.bbox[3] - ob.bbox[1]) <= small_cutoff for ob in obs
            )
            timestamp = obs[0].timestamp_ms
            scored.append((focus_hits * 3 + len(obs) + small, focus_hits, frame, timestamp))

        chosen: list[tuple[int, int, str]] = []  # (frame, timestamp, reason)

        def _try_add(frame: int, timestamp: int, reason: str) -> bool:
            if any(f == frame for f, _, _ in chosen):
                return False
            if any(abs(timestamp - t) < MIN_SPACING_MS for _, t, _ in chosen):
                return False
            chosen.append((frame, timestamp, reason))
            return True

        focus_quota = (per_video + 1) // 2
        for score, focus_hits, frame, timestamp in sorted(
            scored, key=lambda item: (-item[0], item[2])
        ):
            if len(chosen) >= focus_quota:
                break
            _try_add(frame, timestamp, "focus_score")

        all_frames = sorted(by_frame)
        stride_target = per_video - len(chosen)
        if stride_target > 0:
            step = max(1, len(all_frames) // (stride_target + 1))
            for frame in all_frames[step::step]:
                if len(chosen) >= per_video:
                    break
                _try_add(frame, by_frame[frame][0].timestamp_ms, "timeline_stride")
        # 间隔约束可能吃掉配额,按分数补齐
        for score, focus_hits, frame, timestamp in sorted(
            scored, key=lambda item: (-item[0], item[2])
        ):
            if len(chosen) >= per_video:
                break
            _try_add(frame, timestamp, "score_backfill")

        for frame, timestamp, reason in sorted(chosen):
            source = video_dir / "keyframes" / f"kf_{frame:06d}.jpg"
            if not source.exists():
                raise FileNotFoundError(f"keyframe missing: {source}")
            local_name = f"{video_id}_kf_{frame:06d}.jpg"
            shutil.copyfile(source, frames_out / local_name)
            manifest.append(
                {
                    "video_id": video_id,
                    "frame_id": f"{frame:06d}",
                    "frame_index": frame,
                    "timestamp_ms": timestamp,
                    "source_path": str(source),
                    "image": f"frames/{local_name}",
                    "selection_reason": reason,
                    "observation_count": len(by_frame[frame]),
                    "focus_hits": sum(
                        canonical_by_obs.get(ob.observation_id, "") in focus
                        for ob in by_frame[frame]
                    ),
                }
            )
            skeleton_frames.append(
                {"sequence_id": video_id, "frame_id": f"{frame:06d}", "instances": []}
            )

    out = Path(args.out)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "ingest_root": str(root),
                "total_frames": len(manifest),
                "focus_canonicals": sorted(focus),
                "min_spacing_ms": MIN_SPACING_MS,
                "frames": manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "gt.skeleton.json").write_text(
        json.dumps(
            {"dataset_id": args.dataset_id, "frames": skeleton_frames},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"selected_frames": len(manifest), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
