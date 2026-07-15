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

    def detect(self, image_path: str, prompts: list[str]) -> list[RawDetection]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # Grounding DINO 文本约定:小写、句点分隔
        text = ". ".join(p.strip().lower() for p in prompts) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        detections = [
            RawDetection(label=label, score=float(score), box=tuple(float(v) for v in box))
            for label, score, box in zip(results["labels"], results["scores"], results["boxes"])
        ]
        # 确定性输出顺序
        detections.sort(key=lambda d: (-d.score, d.label, d.box))
        return detections
