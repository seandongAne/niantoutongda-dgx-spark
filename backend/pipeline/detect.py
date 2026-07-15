"""开放词汇检测封装(Grounding DINO,transformers 原生)。

只在 Spark 主环境(~/venv,models.yaml env=main)加载;本地测试用
ingest.Detector 协议注入 fake。torch/transformers 延迟导入,导入本模块
本身不需要它们。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDetection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # 像素 x1,y1,x2,y2


class GroundingDinoDetector:
    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.model_version = f"grounding-dino-base@{model_dir}"

    # GDINO 一次喂全量词表会稀释小类得分(2026-07-15 任务A实测:水壶 15 类下
    # 碎片、3 类下 0.56 分)——按批检测再合并,跨批近重框只留高分。
    PROMPT_BATCH_SIZE = 6

    def detect(self, image_path: str, prompts: list[str]) -> list[RawDetection]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        detections: list[RawDetection] = []
        for i in range(0, len(prompts), self.PROMPT_BATCH_SIZE):
            detections += self._detect_batch(image, prompts[i : i + self.PROMPT_BATCH_SIZE])
        detections.sort(key=lambda d: (-d.score, d.label, d.box))
        kept: list[RawDetection] = []
        for d in detections:
            if all(_iou(d.box, k.box) < 0.8 for k in kept):
                kept.append(d)
        return kept

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
