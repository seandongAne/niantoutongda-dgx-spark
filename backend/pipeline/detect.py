"""开放词汇检测封装(Grounding DINO,transformers 原生)。

只在 Spark 主环境(~/venv,models.yaml env=main)加载;本地测试用
ingest.Detector 协议注入 fake。torch/transformers 延迟导入,导入本模块
本身不需要它们。
"""

from __future__ import annotations

import re
from math import ceil
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from backend.pipeline.vocab import CompiledPrompts, normalize_label


@dataclass(frozen=True)
class RawDetection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # 像素 x1,y1,x2,y2
    canonical_id: str | None = None
    category_id: str | None = None
    raw_label: str | None = None


@dataclass(frozen=True)
class _ImageView:
    """One full-frame or tiled view mapped back to its source image."""

    source_index: int
    image: Any
    offset_x: int
    offset_y: int
    score_threshold: float
    is_tile: bool = False
    source_width: int = 0
    source_height: int = 0


def overlapping_tile_boxes(
    width: int,
    height: int,
    *,
    grid: int,
    overlap: float = 0.2,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return deterministic full-coverage ``grid x grid`` overlapping tiles."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if grid < 2:
        raise ValueError("tile grid must be >= 2")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("tile overlap must be in [0, 1)")

    tile_w = min(width, ceil(width / (grid - (grid - 1) * overlap)))
    tile_h = min(height, ceil(height / (grid - (grid - 1) * overlap)))
    max_x, max_y = width - tile_w, height - tile_h
    xs = [round(index * max_x / (grid - 1)) for index in range(grid)]
    ys = [round(index * max_y / (grid - 1)) for index in range(grid)]
    return tuple((x, y, x + tile_w, y + tile_h) for y in ys for x in xs)


def _clutter_score(image: Any) -> float:
    """Cheap deterministic edge-density proxy used to select 3x3 tiles."""

    from PIL import ImageFilter, ImageStat

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0])


def _resolve_label(label: str, mapping: Mapping[str, str]) -> str | None:
    normalized = normalize_label(label)
    normalized_mapping = {normalize_label(alias): value for alias, value in mapping.items()}
    exact = normalized_mapping.get(normalized)
    if exact is not None:
        return exact
    hits = {
        value
        for alias, value in normalized_mapping.items()
        if alias
        and normalized
        and (
            re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized)
            or re.search(rf"(?:^| ){re.escape(normalized)}(?: |$)", alias)
        )
    }
    return next(iter(hits)) if len(hits) == 1 else None


def canonical_aware_nms(
    detections: list[RawDetection],
    *,
    prompt_to_canonical: Mapping[str, str] | None = None,
    prompt_to_category: Mapping[str, str] | None = None,
    iou_threshold: float = 0.8,
) -> list[RawDetection]:
    """Deduplicate only boxes that resolve to the same detector concept.

    Vocab aliases are emitted under their ``canonical_id`` so downstream
    tracklets no longer split by prompt wording.  Unknown/legacy labels use
    their own normalized label as the NMS key; different concepts are retained
    even when their boxes overlap completely.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    canonical_map = prompt_to_canonical or {}
    category_map = prompt_to_category or {}

    prepared: list[tuple[RawDetection, str]] = []
    for detection in detections:
        source_label = detection.raw_label or detection.label
        canonical_id = detection.canonical_id or _resolve_label(source_label, canonical_map)
        category_id = detection.category_id or _resolve_label(source_label, category_map)
        emitted_label = canonical_id or detection.label
        normalized = RawDetection(
            label=emitted_label,
            score=detection.score,
            box=detection.box,
            canonical_id=canonical_id,
            category_id=category_id,
            raw_label=source_label if canonical_id is not None else detection.raw_label,
        )
        nms_key = canonical_id or normalize_label(detection.label)
        prepared.append((normalized, nms_key))

    prepared.sort(
        key=lambda item: (
            -item[0].score,
            item[0].label,
            item[0].box,
            item[0].raw_label or "",
        )
    )
    kept: list[tuple[RawDetection, str]] = []
    for detection, nms_key in prepared:
        if any(
            nms_key == kept_key and _iou(detection.box, existing.box) >= iou_threshold
            for existing, kept_key in kept
        ):
            continue
        kept.append((detection, nms_key))
    return [detection for detection, _ in kept]


class GroundingDinoDetector:
    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.8,
        image_batch_size: int = 8,
        tile_box_threshold: float = 0.22,
        tile_overlap: float = 0.20,
        clutter_tile_count: int = 2,
        tile_max_area_ratio: float = 0.12,
        tile_edge_margin_ratio: float = 0.03,
        tile_max_per_canonical: int = 3,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        if image_batch_size <= 0:
            raise ValueError("image_batch_size must be positive")
        if not 0.0 <= tile_box_threshold <= 1.0:
            raise ValueError("tile_box_threshold must be in [0, 1]")
        if not 0.0 <= tile_overlap < 1.0:
            raise ValueError("tile_overlap must be in [0, 1)")
        if clutter_tile_count < 0:
            raise ValueError("clutter_tile_count cannot be negative")
        if not 0.0 < tile_max_area_ratio <= 1.0:
            raise ValueError("tile_max_area_ratio must be in (0, 1]")
        if not 0.0 <= tile_edge_margin_ratio < 0.5:
            raise ValueError("tile_edge_margin_ratio must be in [0, 0.5)")
        if tile_max_per_canonical <= 0:
            raise ValueError("tile_max_per_canonical must be positive")
        self.image_batch_size = image_batch_size
        self.tile_box_threshold = tile_box_threshold
        self.tile_overlap = tile_overlap
        self.clutter_tile_count = clutter_tile_count
        self.tile_max_area_ratio = tile_max_area_ratio
        self.tile_edge_margin_ratio = tile_edge_margin_ratio
        self.tile_max_per_canonical = tile_max_per_canonical
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        self.nms_iou_threshold = nms_iou_threshold
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.model_version = f"grounding-dino-base@{model_dir}"

    # GDINO 一次喂全量词表会稀释小类得分(2026-07-15 任务A实测:水壶 15 类下
    # 碎片、3 类下 0.56 分;行李箱 6 类批内仍被吞)——按批检测再合并,跨批近重
    # 框只留高分。批次=4;vocab 编译器会把同概念别名及易混概念拆到不同批,
    # 旧 --prompts 路径仍按调用方顺序切批。
    PROMPT_BATCH_SIZE = 4

    def detect(self, image_path: str, prompts: list[str]) -> list[RawDetection]:
        return self.detect_many([image_path], prompts)[0]

    def detect_many(
        self,
        image_paths: Sequence[str],
        prompts: list[str],
        *,
        tiled_image_paths: set[str] | frozenset[str] | None = None,
    ) -> list[list[RawDetection]]:
        """Run frame batching and optional stationary-view tiled detection.

        Every frame always contributes one full-frame view.  Paths selected by
        ingest as >=2s stationary additionally contribute overlapping 2x2
        tiles and the highest-edge-density 3x3 tiles.  All tile boxes are
        translated to full-frame coordinates before canonical-aware NMS.
        """

        from PIL import Image

        if not image_paths:
            return []
        tiled = {str(path) for path in (tiled_image_paths or ())}
        views: list[_ImageView] = []
        for source_index, path in enumerate(image_paths):
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            views.append(
                _ImageView(
                    source_index,
                    image,
                    0,
                    0,
                    self.box_threshold,
                    source_width=width,
                    source_height=height,
                )
            )
            if str(path) not in tiled:
                continue
            for x1, y1, x2, y2 in overlapping_tile_boxes(
                width, height, grid=2, overlap=self.tile_overlap
            ):
                views.append(
                    _ImageView(
                        source_index,
                        image.crop((x1, y1, x2, y2)),
                        x1,
                        y1,
                        self.tile_box_threshold,
                        is_tile=True,
                        source_width=width,
                        source_height=height,
                    )
                )
            if self.clutter_tile_count:
                clutter_candidates = []
                for box in overlapping_tile_boxes(
                    width, height, grid=3, overlap=self.tile_overlap
                ):
                    tile = image.crop(box)
                    clutter_candidates.append((_clutter_score(tile), box, tile))
                clutter_candidates.sort(key=lambda item: (-item[0], item[1]))
                for _, (x1, y1, _, _), tile in clutter_candidates[: self.clutter_tile_count]:
                    views.append(
                        _ImageView(
                            source_index,
                            tile,
                            x1,
                            y1,
                            self.tile_box_threshold,
                            is_tile=True,
                            source_width=width,
                            source_height=height,
                        )
                    )

        if isinstance(prompts, CompiledPrompts):
            batches = prompts.batches
            prompt_to_canonical = prompts.prompt_to_canonical
            prompt_to_category = prompts.prompt_to_category
        else:
            batches = tuple(
                tuple(prompts[i : i + self.PROMPT_BATCH_SIZE])
                for i in range(0, len(prompts), self.PROMPT_BATCH_SIZE)
            )
            prompt_to_canonical = {}
            prompt_to_category = {}

        full_detections_by_source: list[list[RawDetection]] = [[] for _ in image_paths]
        tile_detections_by_source: list[list[RawDetection]] = [[] for _ in image_paths]
        for batch in batches:
            for start in range(0, len(views), self.image_batch_size):
                chunk = views[start : start + self.image_batch_size]
                detected = self._detect_view_batch(chunk, list(batch))
                for view, view_detections in zip(chunk, detected):
                    for detection in view_detections:
                        if detection.score < view.score_threshold:
                            continue
                        x1, y1, x2, y2 = detection.box
                        if view.is_tile:
                            tile_width, tile_height = view.image.size
                            margin_x = tile_width * self.tile_edge_margin_ratio
                            margin_y = tile_height * self.tile_edge_margin_ratio
                            if (
                                x1 <= margin_x
                                or y1 <= margin_y
                                or x2 >= tile_width - margin_x
                                or y2 >= tile_height - margin_y
                            ):
                                continue
                            full_area_ratio = ((x2 - x1) * (y2 - y1)) / max(
                                1.0, float(view.source_width * view.source_height)
                            )
                            if full_area_ratio > self.tile_max_area_ratio:
                                continue
                        mapped = RawDetection(
                            label=detection.label,
                            score=detection.score,
                            box=(
                                x1 + view.offset_x,
                                y1 + view.offset_y,
                                x2 + view.offset_x,
                                y2 + view.offset_y,
                            ),
                            raw_label=detection.raw_label,
                        )
                        target = (
                            tile_detections_by_source
                            if view.is_tile
                            else full_detections_by_source
                        )
                        target[view.source_index].append(mapped)

        merged: list[list[RawDetection]] = []
        for full_detections, tile_detections in zip(
            full_detections_by_source, tile_detections_by_source
        ):
            full = canonical_aware_nms(
                full_detections,
                prompt_to_canonical=prompt_to_canonical,
                prompt_to_category=prompt_to_category,
                iou_threshold=self.nms_iou_threshold,
            )
            tile = canonical_aware_nms(
                tile_detections,
                prompt_to_canonical=prompt_to_canonical,
                prompt_to_category=prompt_to_category,
                iou_threshold=self.nms_iou_threshold,
            )
            tile.sort(
                key=lambda item: (
                    item.canonical_id or normalize_label(item.label),
                    -item.score,
                    item.box,
                )
            )
            limited_tile: list[RawDetection] = []
            counts: dict[str, int] = {}
            for detection in tile:
                key = detection.canonical_id or normalize_label(detection.label)
                if counts.get(key, 0) >= self.tile_max_per_canonical:
                    continue
                counts[key] = counts.get(key, 0) + 1
                limited_tile.append(detection)
            merged.append(
                canonical_aware_nms(
                    full + limited_tile,
                    prompt_to_canonical=prompt_to_canonical,
                    prompt_to_category=prompt_to_category,
                    iou_threshold=self.nms_iou_threshold,
                )
            )
        return merged

    def _detect_view_batch(
        self, views: Sequence[_ImageView], prompts: list[str]
    ) -> list[list[RawDetection]]:
        # Grounding DINO 文本约定:小写、句点分隔
        text = ". ".join(p.strip().lower() for p in prompts) + "."
        images = [view.image for view in views]
        inputs = self.processor(
            images=images,
            text=[text] * len(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # 先用低阈值解码，随后按 full/tile 各自阈值过滤。
            threshold=min(self.box_threshold, self.tile_box_threshold),
            text_threshold=self.text_threshold,
            target_sizes=[view.image.size[::-1] for view in views],
        )
        detected: list[list[RawDetection]] = []
        for result in results:
            labels = result.get(
                "text_labels", result["labels"]
            )  # 新版文本标签迁到 text_labels
            detected.append(
                [
                    RawDetection(
                        label=str(label),
                        score=float(score),
                        box=tuple(float(value) for value in box),
                    )
                    for label, score, box in zip(labels, result["scores"], result["boxes"])
                ]
            )
        return detected


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
