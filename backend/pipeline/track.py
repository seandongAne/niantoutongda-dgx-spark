"""S2 单视频轨迹 — 确定性贪心 IoU 关联(ByteTrack 简化型,无权重)。

确定性保证:候选对按 (iou 降序, track_id 升序, det 序号升序) 排序后贪心,
新轨迹按检测序号顺序分配 id;同输入必同输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    timestamp_ms: int
    box: Box
    label: str
    score: float
    ref: str = ""  # 上游 Observation id 等外部引用,追踪器本身不使用
    hero_score: float = 0.0  # 面积 x 清晰度 x 完整度；只用于证据帧排序


@dataclass
class Track:
    track_id: int
    label: str
    detections: list[FrameDetection] = field(default_factory=list)
    last_frame_index: int = -1
    missed: int = 0  # 连续未匹配的关键帧数


class GreedyIoUTracker:
    """逐关键帧调用 update();全部帧喂完后 finalize() 取轨迹。

    同标签才可关联(开放词汇标签来自检测器,跨视频身份由 S3 决定,
    这里只做视频内的时间连续性)。
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 2, min_track_len: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_track_len = min_track_len
        self._active: list[Track] = []
        self._finished: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[FrameDetection]) -> None:
        candidates = [
            (iou(t.detections[-1].box, d.box), t.track_id, di, t, d)
            for t in self._active
            for di, d in enumerate(detections)
            if t.label == d.label
        ]
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for ov, track_id, di, track, det in candidates:
            if ov < self.iou_threshold:
                break
            if track_id in matched_tracks or di in matched_dets:
                continue
            track.detections.append(det)
            track.last_frame_index = det.frame_index
            track.missed = 0
            matched_tracks.add(track_id)
            matched_dets.add(di)

        survivors: list[Track] = []
        for t in self._active:
            if t.track_id in matched_tracks:
                survivors.append(t)
                continue
            t.missed += 1
            if t.missed > self.max_missed:
                self._finished.append(t)
            else:
                survivors.append(t)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            survivors.append(
                Track(
                    track_id=self._next_id,
                    label=det.label,
                    detections=[det],
                    last_frame_index=det.frame_index,
                )
            )
            self._next_id += 1

        self._active = survivors

    def finalize(self) -> list[Track]:
        tracks = self._finished + self._active
        self._active, self._finished = [], []
        tracks = [t for t in tracks if len(t.detections) >= self.min_track_len]
        tracks.sort(key=lambda t: t.track_id)
        return tracks
