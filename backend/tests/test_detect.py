"""Alias/canonical-aware cross-batch NMS tests (no torch required)."""

from backend.pipeline.detect import (
    GroundingDinoDetector,
    RawDetection,
    canonical_aware_nms,
    overlapping_tile_boxes,
)


BOX = (0.0, 0.0, 100.0, 100.0)
OVERLAP_09 = (0.0, 0.0, 100.0, 90.0)


def test_high_overlap_different_canonical_concepts_are_both_kept():
    detections = [
        RawDetection("security camera", 0.90, BOX),
        RawDetection("smart speaker", 0.85, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "security camera": "security_camera",
            "smart speaker": "night_light",
        },
    )

    assert len(kept) == 2
    assert {d.label for d in kept} == {"security_camera", "night_light"}


def test_high_overlap_aliases_of_same_canonical_keep_higher_score():
    detections = [
        RawDetection("smart speaker", 0.72, BOX),
        RawDetection("cylinder lamp", 0.91, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "smart speaker": "night_light",
            "cylinder lamp": "night_light",
        },
        prompt_to_category={"smart speaker": "lamp", "cylinder lamp": "lamp"},
    )

    assert kept == [
        RawDetection(
            label="night_light",
            score=0.91,
            box=OVERLAP_09,
            canonical_id="night_light",
            category_id="lamp",
            raw_label="cylinder lamp",
        )
    ]


def test_same_semantic_category_does_not_trigger_nms_across_canonicals():
    detections = [
        RawDetection("smart speaker", 0.90, BOX),
        RawDetection("table lamp", 0.89, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "smart speaker": "night_light",
            "table lamp": "table_lamp",
        },
        prompt_to_category={"smart speaker": "lamp", "table lamp": "lamp"},
    )

    assert len(kept) == 2
    assert {d.canonical_id for d in kept} == {"night_light", "table_lamp"}


def test_unique_truncated_text_label_resolves_but_ambiguous_one_does_not():
    kept = canonical_aware_nms(
        [RawDetection("camera", 0.9, BOX), RawDetection("lamp", 0.8, OVERLAP_09)],
        prompt_to_canonical={
            "security camera": "security_camera",
            "cylinder lamp": "night_light",
            "table lamp": "table_lamp",
        },
        prompt_to_category={
            "security camera": "camera",
            "cylinder lamp": "lamp",
            "table lamp": "lamp",
        },
    )

    assert kept[0].canonical_id == "security_camera"
    assert kept[1].canonical_id is None


def test_overlapping_tiles_cover_frame_and_are_deterministic():
    boxes = overlapping_tile_boxes(100, 80, grid=2, overlap=0.2)

    assert boxes == overlapping_tile_boxes(100, 80, grid=2, overlap=0.2)
    assert len(boxes) == 4
    assert boxes[0][:2] == (0, 0)
    assert boxes[-1][2:] == (100, 80)
    left, right = boxes[0], boxes[1]
    assert left[2] > right[0]


def test_frame_batch_results_equal_single_frame_path(tmp_path):
    from PIL import Image

    paths = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (100, 80), color=(index * 20, 0, 0)).save(path)
        paths.append(str(path))

    detector = object.__new__(GroundingDinoDetector)
    detector.box_threshold = 0.3
    detector.tile_box_threshold = 0.22
    detector.tile_overlap = 0.2
    detector.clutter_tile_count = 0
    detector.tile_max_area_ratio = 0.12
    detector.tile_edge_margin_ratio = 0.03
    detector.tile_max_per_canonical = 3
    detector.image_batch_size = 2
    detector.nms_iou_threshold = 0.8

    def fake_detect_view_batch(views, prompts):
        return [
            [RawDetection(label=prompts[0], score=0.9, box=(10.0, 10.0, 40.0, 40.0))]
            for _ in views
        ]

    detector._detect_view_batch = fake_detect_view_batch

    sequential = [detector.detect(path, ["desk"]) for path in paths]
    batched = detector.detect_many(paths, ["desk"])
    assert batched == sequential


def test_tiled_path_rejects_edge_large_boxes_and_caps_each_canonical(tmp_path):
    from PIL import Image

    path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(path)
    detector = object.__new__(GroundingDinoDetector)
    detector.box_threshold = 0.3
    detector.tile_box_threshold = 0.22
    detector.tile_overlap = 0.2
    detector.clutter_tile_count = 0
    detector.image_batch_size = 8
    detector.nms_iou_threshold = 0.8
    detector.tile_max_area_ratio = 0.12
    detector.tile_edge_margin_ratio = 0.03
    detector.tile_max_per_canonical = 2

    def fake_detect_view_batch(views, prompts):
        rows = []
        for view in views:
            if not view.is_tile:
                rows.append([])
                continue
            width, height = view.image.size
            rows.append(
                [
                    RawDetection(prompts[0], 0.90, (10.0, 10.0, 25.0, 25.0)),
                    RawDetection(prompts[0], 0.89, (0.0, 0.0, 12.0, 12.0)),
                    RawDetection(
                        prompts[0],
                        0.88,
                        (5.0, 5.0, float(width - 5), float(height - 5)),
                    ),
                ]
            )
        return rows

    detector._detect_view_batch = fake_detect_view_batch
    detected = detector.detect_many([str(path)], ["water bottle"], tiled_image_paths={str(path)})[0]

    assert 1 <= len(detected) <= 2
    assert all(item.box[0] > 0 and item.box[1] > 0 for item in detected)
    assert all((item.box[2] - item.box[0]) * (item.box[3] - item.box[1]) <= 0.12 * 100 * 80 for item in detected)
