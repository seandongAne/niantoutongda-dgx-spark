"""实例嵌入封装(DINOv2 冻结骨干,CLS token,L2 归一化)。

SF1-L1 的投影头训练在此向量之上;本模块只负责骨干前向。
只在 Spark 主环境加载,延迟导入。
"""

from __future__ import annotations


class Dinov2Embedder:
    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.model_version = f"dinov2-base@{model_dir}"

    def embed(self, image_path: str) -> list[float]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[0, 0]
        cls = cls / cls.norm().clamp_min(1e-12)
        return [float(v) for v in cls.cpu()]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))
