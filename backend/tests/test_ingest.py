"""S1+S2 端到端(合成视频 + fake 检测器/嵌入器,不需要 torch)。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline.detect import RawDetection
from backend.pipeline.ingest import ingest_video
from backend.pipeline.keyframes import sample_keyframes
from backend.schemas.core import Observation, Tracklet

FRAME_W, FRAME_H, FPS, N_FRAMES = 320, 240, 10.0, 40


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """白底上一个右移的红方块 + 一个静止的蓝方块。"""
    path = tmp_path_factory.mktemp("video") / "synth.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (FRAME_W, FRAME_H)
    )
    for f in range(N_FRAMES):
        frame = np.full((FRAME_H, FRAME_W, 3), 255, np.uint8)
        x = 10 + f * 2
        frame[60:120, x : x + 50] = (0, 0, 255)  # 移动红块(BGR)
        frame[150:200, 220:270] = (255, 0, 0)  # 静止蓝块
        writer.write(frame)
    writer.release()
    return str(path)


class FakeDetector:
    """从像素直接找色块,模拟开放词汇检测器。"""

    model_version = "fake-color-detector@test"

    def detect(self, image_path, prompts):
        img = cv2.imread(image_path)
        out = []
        for label, mask in (
            ("red box", (img[:, :, 2] > 200) & (img[:, :, 0] < 80)),
            ("blue box", (img[:, :, 0] > 200) & (img[:, :, 2] < 80)),
        ):
            ys, xs = np.nonzero(mask)
            if len(xs) < 50:
                continue
            out.append(
                RawDetection(
                    label=label,
                    score=0.9,
                    box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                )
            )
        return out


class FakeEmbedder:
    model_version = "fake-embedder@test"

    def embed(self, image_path):
        img = cv2.imread(image_path)
        v = [float(img[:, :, c].mean()) for c in range(3)]
        n = max(sum(x * x for x in v) ** 0.5, 1e-12)
        return [x / n for x in v]


def test_keyframes_dedup_static_video(tmp_path):
    path = tmp_path / "static.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 64))
    frame = np.full((64, 64, 3), 128, np.uint8)
    for _ in range(30):
        writer.write(frame)
    writer.release()
    kfs = sample_keyframes(path, tmp_path / "kf", target_fps=5.0)
    assert len(kfs) == 1  # 全静止只留第一帧


def test_ingest_end_to_end(synthetic_video, tmp_path):
    result = ingest_video(
        video_id="v_test",
        video_path=synthetic_video,
        prompts=["red box", "blue box"],
        workdir=tmp_path / "run",
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        config_version="test-v1",
    )
    # 两个物体 → 两条轨迹,移动不裂轨
    assert len(result.tracklets) == 2
    labels = {t.attributes["label"] for t in result.tracklets}
    assert labels == {"red box", "blue box"}

    # 每条轨迹有证据裁剪 + 嵌入引用,且文件真实存在
    for t in result.tracklets:
        assert 1 <= len(t.prototype_refs) <= 3
        for p in t.prototype_refs:
            assert Path(p).exists()
        assert t.embedding_ref and Path(t.embedding_ref).exists()
        vec = json.loads(Path(t.embedding_ref).read_text())["vector"]
        assert abs(sum(v * v for v in vec) - 1.0) < 1e-6  # 归一化

    # 轨迹引用的 observation 全部存在且属于该视频
    obs_ids = {o.observation_id for o in result.observations}
    for t in result.tracklets:
        assert set(t.observation_ids) <= obs_ids

    # 产物文件符合契约(逐行可解析)
    workdir = Path(result.workdir)
    obs_lines = (workdir / "observations.jsonl").read_text().strip().splitlines()
    trk_lines = (workdir / "tracklets.jsonl").read_text().strip().splitlines()
    assert len(obs_lines) == len(result.observations)
    for line in obs_lines:
        Observation.model_validate_json(line)
    for line in trk_lines:
        Tracklet.model_validate_json(line)

    # 审计链:三个阶段事件都在
    audit_lines = (workdir / "audit-events.jsonl").read_text().strip().splitlines()
    events = [json.loads(line) for line in audit_lines]
    assert [e["event_type"] for e in events] == [
        "KeyframesSampled",
        "DetectionCompleted",
        "TrackletsFormed",
    ]


def test_ingest_deterministic(synthetic_video, tmp_path):
    runs = []
    for i in range(2):
        r = ingest_video(
            video_id="v_test",
            video_path=synthetic_video,
            prompts=["red box", "blue box"],
            workdir=tmp_path / f"run{i}",
            detector=FakeDetector(),
            config_version="test-v1",
        )
        runs.append(
            [(t.tracklet_id, t.attributes["label"], tuple(t.observation_ids)) for t in r.tracklets]
        )
    assert runs[0] == runs[1]
