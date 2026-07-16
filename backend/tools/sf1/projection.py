"""不依赖 Torch 的两层投影头推理与可校验权重格式。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = "sf1-projection-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NumpyProjectionHead:
    """Linear(input→hidden) + ReLU + Linear(hidden→output) + L2 normalize。"""

    weight1: np.ndarray  # [hidden, input]
    bias1: np.ndarray  # [hidden]
    weight2: np.ndarray  # [output, hidden]
    bias2: np.ndarray  # [output]

    def __post_init__(self) -> None:
        arrays = (self.weight1, self.bias1, self.weight2, self.bias2)
        if any(array.dtype != np.float32 for array in arrays):
            raise ValueError("projection arrays must be float32")
        if self.weight1.ndim != 2 or self.weight2.ndim != 2:
            raise ValueError("projection weights must be matrices")
        if self.bias1.shape != (self.weight1.shape[0],):
            raise ValueError("bias1 shape mismatch")
        if self.weight2.shape[1] != self.weight1.shape[0]:
            raise ValueError("hidden dimension mismatch")
        if self.bias2.shape != (self.weight2.shape[0],):
            raise ValueError("bias2 shape mismatch")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("projection contains non-finite values")

    @property
    def input_dim(self) -> int:
        return int(self.weight1.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.weight1.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.weight2.shape[0])

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        was_vector = values.ndim == 1
        if was_vector:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [N,{self.input_dim}] vectors, got {tuple(values.shape)}"
            )
        hidden = np.maximum(values @ self.weight1.T + self.bias1, 0.0)
        output = hidden @ self.weight2.T + self.bias2
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        if (norms < 1e-12).any() or not np.isfinite(norms).all():
            raise ValueError("projection produced zero or non-finite vector")
        output = (output / norms).astype(np.float32, copy=False)
        return output[0] if was_vector else output

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            format_version=np.asarray(FORMAT_VERSION),
            weight1=self.weight1,
            bias1=self.bias1,
            weight2=self.weight2,
            bias2=self.bias2,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "NumpyProjectionHead":
        target = Path(path)
        if expected_sha256:
            actual = sha256_file(target)
            if actual != expected_sha256:
                raise ValueError(
                    f"projection sha256 mismatch: expected {expected_sha256}, got {actual}"
                )
        with np.load(target, allow_pickle=False) as data:
            version = str(data["format_version"].item())
            if version != FORMAT_VERSION:
                raise ValueError(f"unsupported projection format: {version}")
            return cls(
                weight1=np.asarray(data["weight1"], dtype=np.float32),
                bias1=np.asarray(data["bias1"], dtype=np.float32),
                weight2=np.asarray(data["weight2"], dtype=np.float32),
                bias2=np.asarray(data["bias2"], dtype=np.float32),
            )
