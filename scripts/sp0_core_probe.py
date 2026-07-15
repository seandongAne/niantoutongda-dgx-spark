#!/usr/bin/env python
"""SP0-core 探针 — 在 Spark 主环境(~/venv)运行,阻塞 S1 的生死门。

用法: python scripts/sp0_core_probe.py [--image 真实图片路径] [--run-id sp0core_YYYYMMDD]
产出: results/acceptance/sp0/<run_id>/metrics.json + 探针输出图

无 --image 时用 PIL 合成图:此时检测/嵌入探针只证明"加载 + 前向 + 显存",
不证明真实场景质量——真实输出探针等 G0 素材就位后用同一脚本重跑。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

MODELS = Path.home() / "models"
DETECTOR_DIR = str(MODELS / "IDEA-Research__grounding-dino-base")
EMBEDDER_DIR = str(MODELS / "facebook__dinov2-base")


def probe_torch(metrics: dict) -> bool:
    import torch

    t0 = time.perf_counter()
    ok = torch.cuda.is_available()
    metrics["torch"] = {"version": torch.__version__, "cuda_available": ok}
    if not ok:
        return False
    x = torch.rand(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    metrics["torch"].update(
        device=torch.cuda.get_device_name(0),
        matmul_4096_s=round(time.perf_counter() - t0, 3),
        result_shape=list(y.shape),
    )
    return True


def synth_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([80, 120, 240, 360], fill=(180, 40, 40))
    d.rectangle([380, 200, 560, 420], fill=(40, 60, 180))
    d.ellipse([280, 60, 360, 140], fill=(240, 200, 40))
    img.save(path)


def peak_mem_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("sp0core_%Y%m%d_%H%M")
    out_dir = PROJ / "results" / "acceptance" / "sp0" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_mode": "real" if args.image else "synthetic_load_probe_only",
    }

    failures: list[str] = []

    if not probe_torch(metrics):
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("SP0CORE_FAIL torch_cuda_unavailable")
        return 1

    image = args.image
    if image is None:
        image = str(out_dir / "synth_probe.png")
        synth_image(Path(image))

    import torch

    # 检测探针
    try:
        torch.cuda.reset_peak_memory_stats()
        from backend.pipeline.detect import GroundingDinoDetector

        t0 = time.perf_counter()
        det = GroundingDinoDetector(DETECTOR_DIR)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        detections = det.detect(image, ["red box", "blue box", "lamp", "mug"])
        metrics["detection"] = {
            "model": det.model_version,
            "load_s": round(load_s, 2),
            "infer_s": round(time.perf_counter() - t0, 2),
            "peak_mem_gb": peak_mem_gb(),
            "num_detections": len(detections),
            "top": [
                {"label": d.label, "score": round(d.score, 3), "box": [round(v, 1) for v in d.box]}
                for d in detections[:5]
            ],
        }
        del det
        torch.cuda.empty_cache()
    except Exception as e:  # 探针必须报告失败而不是崩掉整个脚本
        failures.append(f"detection: {type(e).__name__}: {e}")
        metrics["detection"] = {"error": str(e)}

    # 嵌入探针(同图两次 → 余弦应≈1;真实困难负样本等 G0)
    try:
        torch.cuda.reset_peak_memory_stats()
        from backend.pipeline.embed import Dinov2Embedder

        t0 = time.perf_counter()
        emb = Dinov2Embedder(EMBEDDER_DIR)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        v1 = emb.embed(image)
        v2 = emb.embed(image)
        metrics["embedding"] = {
            "model": emb.model_version,
            "load_s": round(load_s, 2),
            "infer_s_per_image": round((time.perf_counter() - t0) / 2, 3),
            "peak_mem_gb": peak_mem_gb(),
            "dim": len(v1),
            "self_cosine": round(Dinov2Embedder.cosine(v1, v2), 6),
        }
        del emb
        torch.cuda.empty_cache()
    except Exception as e:
        failures.append(f"embedding: {type(e).__name__}: {e}")
        metrics["embedding"] = {"error": str(e)}

    # CP-SAT 冒烟(OPTIMAL + INFEASIBLE 双态)
    try:
        from backend.tools.solver.layout_solver import (
            CandidateRegion,
            LayoutProblem,
            PlacementUnit,
            solve_layout,
        )

        ok_problem = LayoutProblem(
            units=[PlacementUnit("g1", 1, False, frozenset({"surface"}))],
            regions=[CandidateRegion("r1", "surface", 2, True)],
            scores={("g1", "r1"): 5},
        )
        bad_problem = LayoutProblem(
            units=[PlacementUnit("g1", 1, True, frozenset({"surface"}))],
            regions=[CandidateRegion("r1", "surface", 2, False)],  # 无电源证据
            scores={("g1", "r1"): 5},
        )
        metrics["cpsat"] = {
            "optimal_status": solve_layout(ok_problem).status,
            "infeasible_status": solve_layout(bad_problem).status,
        }
    except Exception as e:
        failures.append(f"cpsat: {type(e).__name__}: {e}")
        metrics["cpsat"] = {"error": str(e)}

    metrics["failures"] = failures
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("SP0CORE_FAIL " + "; ".join(failures) if failures else "SP0CORE_PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
