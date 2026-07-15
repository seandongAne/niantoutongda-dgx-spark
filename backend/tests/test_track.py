"""确定性贪心 IoU 追踪器测试。"""

from backend.pipeline.track import Box, FrameDetection, GreedyIoUTracker, iou


def _det(frame, x, label="lamp", score=0.9, size=100.0):
    return FrameDetection(
        frame_index=frame,
        timestamp_ms=frame * 500,
        box=Box(x, 0.0, x + size, size),
        label=label,
        score=score,
    )


def test_iou_basic():
    a = Box(0, 0, 100, 100)
    assert iou(a, a) == 1.0
    assert iou(a, Box(200, 200, 300, 300)) == 0.0
    assert abs(iou(a, Box(50, 0, 150, 100)) - 1 / 3) < 1e-9


def test_moving_object_stays_one_track():
    tracker = GreedyIoUTracker()
    for f in range(5):
        tracker.update([_det(f, x=f * 20.0)])  # 每帧右移 20px,IoU 仍高
    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 5


def test_two_labels_never_merge():
    tracker = GreedyIoUTracker()
    for f in range(3):
        tracker.update([_det(f, x=0.0, label="lamp"), _det(f, x=10.0, label="mug")])
    tracks = tracker.finalize()
    assert len(tracks) == 2
    assert {t.label for t in tracks} == {"lamp", "mug"}


def test_gap_within_max_missed_survives():
    tracker = GreedyIoUTracker(max_missed=2)
    tracker.update([_det(0, x=0.0)])
    tracker.update([])  # 丢 1 帧
    tracker.update([])  # 丢 2 帧
    tracker.update([_det(3, x=0.0)])
    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 2


def test_long_gap_spawns_new_track():
    tracker = GreedyIoUTracker(max_missed=1, min_track_len=1)
    tracker.update([_det(0, x=0.0)])
    tracker.update([])
    tracker.update([])  # 超过 max_missed,轨迹退役
    tracker.update([_det(3, x=0.0)])
    tracks = tracker.finalize()
    assert len(tracks) == 2


def test_min_track_len_filters_flicker():
    tracker = GreedyIoUTracker(min_track_len=2)
    tracker.update([_det(0, x=0.0), _det(0, x=500.0)])  # 第二个只出现一帧
    tracker.update([_det(1, x=0.0)])
    tracker.update([_det(2, x=0.0)])
    tracks = tracker.finalize()
    assert len(tracks) == 1


def test_deterministic_ids_and_greedy_by_iou():
    def run():
        tracker = GreedyIoUTracker()
        tracker.update([_det(0, x=0.0), _det(0, x=60.0)])  # 两框有重叠
        tracker.update([_det(1, x=5.0), _det(1, x=65.0)])
        return [(t.track_id, [d.box.x1 for d in t.detections]) for t in tracker.finalize()]

    first, second = run(), run()
    assert first == second  # 同输入同输出
    assert first == [(1, [0.0, 5.0]), (2, [60.0, 65.0])]  # 各自跟最近的框
