"""开放词汇检测封装(Grounding DINO,transformers 原生)。

只在 Spark 主环境(~/venv,models.yaml env=main)加载;本地测试用
ingest.Detector 协议注入 fake。torch/transformers 延迟导入,导入本模块
本身不需要它们。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from backend.pipeline.vocab import CompiledPrompts, normalize_label


@dataclass(frozen=True)
class RawDetection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # 像素 x1,y1,x2,y2
    canonical_id: str | None = None
    category_id: str | None = None
    raw_label: str | None = None


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
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
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
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        detections: list[RawDetection] = []
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
        for batch in batches:
            detections += self._detect_batch(image, list(batch))
        return canonical_aware_nms(
            detections,
            prompt_to_canonical=prompt_to_canonical,
            prompt_to_category=prompt_to_category,
            iou_threshold=self.nms_iou_threshold,
        )

    def _detect_batch(self, image, prompts: list[str]) -> list[RawDetection]:
        # Grounding DINO 文本约定:小写、句点分隔
        text = ". ".join(p.strip().lower() for p in prompts) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,  # transformers 4.51+ 改名(原 box_threshold)
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = results.get("text_labels", results["labels"])  # 新版文本标签迁到 text_labels
        return [
            RawDetection(label=label, score=float(score), box=tuple(float(v) for v in box))
            for label, score, box in zip(labels, results["scores"], results["boxes"])
        ]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
