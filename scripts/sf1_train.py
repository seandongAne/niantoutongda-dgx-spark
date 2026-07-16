#!/usr/bin/env python
"""SF1-L1：冻结 DINOv2 上训练轻量两层投影头并产出可接线权重。

训练只读取已生成的 768d tracklet 嵌入，不重新加载 DINOv2。正式长任务请按
项目纪律在 Spark 上 nohup 发射；产物目录可由 pull_results.sh 小体积拉回。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# torch deterministic matmul on CUDA requires this to be set before CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.sf1.dataset import (  # noqa: E402
    build_leave_last_video_out_split,
    dataset_fingerprint,
    load_labeled_samples,
)
from backend.tools.sf1.metrics import retrieval_metrics  # noqa: E402
from backend.tools.sf1.projection import (  # noqa: E402
    NumpyProjectionHead,
    sha256_file,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        stamp = PROJ / "COMMIT"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


def _sample_arrays(samples, identity_index: dict[str, int]):
    vectors = np.stack([sample.vector for sample in samples]).astype(np.float32)
    labels = np.asarray(
        [identity_index[sample.identity_id] for sample in samples], dtype=np.int64
    )
    return vectors, labels


def _head_from_torch(model, cfg: dict) -> NumpyProjectionHead:
    return NumpyProjectionHead(
        weight1=model.first.weight.detach().cpu().numpy().astype(np.float32),
        bias1=model.first.bias.detach().cpu().numpy().astype(np.float32),
        weight2=model.second.weight.detach().cpu().numpy().astype(np.float32),
        bias2=model.second.bias.detach().cpu().numpy().astype(np.float32),
        mode=str(cfg.get("mode", "plain")),
        residual_scale=float(cfg.get("residual_scale", 1.0)),
    )


def _train_one(cfg: dict, split, seed: int, device: str, baseline: dict):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("SF1 training requires PyTorch (run in Spark main env)") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    torch.use_deterministic_algorithms(True)

    identities = sorted({sample.identity_id for sample in split.train})
    identity_index = {identity: index for index, identity in enumerate(identities)}
    train_vectors, train_labels = _sample_arrays(split.train, identity_index)
    x = torch.from_numpy(train_vectors).to(device)
    labels = torch.from_numpy(train_labels).to(device)

    class ProjectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = str(cfg.get("mode", "plain"))
            self.residual_scale = float(cfg.get("residual_scale", 1.0))
            self.first = nn.Linear(int(cfg["input_dim"]), int(cfg["hidden_dim"]))
            self.second = nn.Linear(int(cfg["hidden_dim"]), int(cfg["output_dim"]))
            if self.mode == "residual":
                if int(cfg["output_dim"]) != int(cfg["input_dim"]):
                    raise ValueError("residual mode requires output_dim == input_dim")
                # 恒等起点：epoch 0 与原始 DINO 向量逐字节等价（归一化误差除外）。
                nn.init.zeros_(self.second.weight)
                nn.init.zeros_(self.second.bias)

        def forward(self, values):
            learned = self.second(F.relu(self.first(values)))
            if self.mode == "residual":
                return values + self.residual_scale * learned
            return learned

    model = ProjectionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    temperature = float(cfg["temperature"])
    eval_every = int(cfg.get("eval_every", 5))
    history = []
    train_matrix = np.stack([sample.vector for sample in split.train])
    validation_matrix = np.stack([sample.vector for sample in split.validation])
    best_state = None
    best_metrics = None
    best_epoch = None
    best_score = None
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        z = F.normalize(model(x), dim=1)
        logits = (z @ z.T) / temperature
        eye = torch.eye(len(labels), dtype=torch.bool, device=device)
        positives = labels[:, None].eq(labels[None, :]) & ~eye
        candidates = ~eye
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * candidates
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positives.sum(dim=1)
        valid = positive_count > 0
        loss = -(
            (log_prob * positives).sum(dim=1)[valid]
            / positive_count[valid]
        ).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"seed {seed} epoch {epoch}: non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        should_eval = epoch % eval_every == 0 or epoch == int(cfg["epochs"])
        if should_eval:
            head = _head_from_torch(model, cfg)
            metrics = retrieval_metrics(
                split.train,
                split.validation,
                projected_train=head.apply(train_matrix),
                projected_validation=head.apply(validation_matrix),
            )
            eligible = (
                metrics["recall_at_1"] >= baseline["recall_at_1"]
                and metrics["recall_at_5"] >= baseline["recall_at_5"]
            )
            # 选择策略在 config 冻结：先守住 R@1/R@5，再最大化平均 top-1 margin。
            score = (
                float(metrics["top1_margin_mean"]),
                float(metrics["mean_reciprocal_rank"]),
            )
            if eligible and (best_score is None or score > best_score):
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                best_epoch = epoch
            history.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 8),
                    "eligible": eligible,
                    "metrics": metrics,
                }
            )

    if best_state is None:
        # 不把退化权重包装成成功；保留最后权重与失败 gate 供诊断。
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = int(cfg["epochs"])
    model.load_state_dict(best_state)
    head = _head_from_torch(model, cfg)
    metrics = best_metrics or retrieval_metrics(
        split.train,
        split.validation,
        projected_train=head.apply(train_matrix),
        projected_validation=head.apply(validation_matrix),
    )
    return head, metrics, history, torch.__version__, best_epoch, best_metrics is not None


def _stability(per_seed: list[dict]) -> dict:
    keys = (
        "recall_at_1",
        "recall_at_5",
        "mean_reciprocal_rank",
        "positive_cosine_mean",
        "hardest_negative_cosine_mean",
        "top1_margin_mean",
    )
    summary = {}
    for key in keys:
        values = np.asarray([item["metrics"][key] for item in per_seed], dtype=float)
        summary[key] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std(ddof=0)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-root", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg.get("split_policy") != "leave_last_video_out_v1":
        raise ValueError(f"unsupported split_policy: {cfg.get('split_policy')}")
    samples = load_labeled_samples(
        args.ingest_root,
        args.labels,
        input_dim=int(cfg["input_dim"]),
    )
    split = build_leave_last_video_out_split(samples)
    baseline = retrieval_metrics(split.train, split.validation)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SF1 training requires PyTorch (run in Spark main env)") from exc
    requested_device = args.device or str(cfg.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requires cuda but torch.cuda.is_available() is false")
    device = requested_device
    started = time.monotonic()
    per_seed = []
    primary_head = None
    all_history = []
    torch_version = "unknown"
    for index, seed in enumerate(int(value) for value in cfg["seeds"]):
        head, metrics, history, torch_version, selected_epoch, eligible = _train_one(
            cfg, split, seed, device, baseline
        )
        if index == 0:
            primary_head = head
        per_seed.append(
            {
                "seed": seed,
                "selected_epoch": selected_epoch,
                "eligible_checkpoint_found": eligible,
                "metrics": metrics,
            }
        )
        all_history.extend(history)
        print(
            json.dumps(
                {"seed": seed, "selected_epoch": selected_epoch,
                 "recall_at_1": metrics["recall_at_1"],
                 "recall_at_5": metrics["recall_at_5"],
                 "top1_margin_mean": metrics["top1_margin_mean"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    assert primary_head is not None

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "projection.npz"
    primary_head.save(artifact)
    artifact_sha = sha256_file(artifact)
    split_manifest = split.manifest()
    split_manifest.update(
        {
            "dataset_fingerprint": dataset_fingerprint(samples),
            "labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
            "ingest_root": str(args.ingest_root),
        }
    )
    (out / "split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary = per_seed[0]["metrics"]
    deltas = {
        key: round(float(primary[key]) - float(baseline[key]), 6)
        for key in (
            "recall_at_1",
            "recall_at_5",
            "mean_reciprocal_rank",
            "positive_cosine_mean",
            "hardest_negative_cosine_mean",
            "top1_margin_mean",
        )
    }
    gate = {
        "finite": bool(primary["finite"]),
        "learned_checkpoint_found": bool(
            per_seed[0]["eligible_checkpoint_found"]
            and per_seed[0]["selected_epoch"] > 0
        ),
        "recall_at_1_non_degrading": primary["recall_at_1"] >= baseline["recall_at_1"],
        "recall_at_5_non_degrading": primary["recall_at_5"] >= baseline["recall_at_5"],
        "top1_margin_improved": primary["top1_margin_mean"] > baseline["top1_margin_mean"],
    }
    gate["pass"] = all(gate.values())
    metrics_payload = {
        "scope": cfg["scope"],
        "selection_policy": cfg["selection_policy"],
        "baseline": baseline,
        "primary_seed": per_seed[0],
        "deltas": deltas,
        "per_seed": per_seed,
        "stability": _stability(per_seed),
        "gate": gate,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "training-history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_history),
        encoding="utf-8",
    )

    runtime = {
        "device": device,
        "torch_version": torch_version,
        "cuda_device": (
            torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "slice_id": "SF1-L1",
        "version": cfg["version"],
        "scope": cfg["scope"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
        "dataset_fingerprint": dataset_fingerprint(samples),
        "split_policy": split.policy,
        "selection_policy": cfg["selection_policy"],
        "primary_seed": int(cfg["seeds"][0]),
        "selected_epoch": int(per_seed[0]["selected_epoch"]),
        "projection": {
            "path": str(artifact),
            "sha256": artifact_sha,
            "format": "sf1-projection-v1",
            "input_dim": primary_head.input_dim,
            "hidden_dim": primary_head.hidden_dim,
            "output_dim": primary_head.output_dim,
            "mode": primary_head.mode,
            "residual_scale": primary_head.residual_scale,
            "parameters": (
                primary_head.weight1.size
                + primary_head.bias1.size
                + primary_head.weight2.size
                + primary_head.bias2.size
            ),
        },
        "runtime": runtime,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gate": gate,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "projection.yaml").write_text(
        "projection:\n"
        "  enabled: true\n"
        f"  artifact: {artifact}\n"
        f"  sha256: {artifact_sha}\n",
        encoding="utf-8",
    )
    (out / "failure-case.md").write_text(
        "# SF1-L1 证据边界\n\n"
        f"- scope: `{cfg['scope']}`\n"
        "- dev_a 结果只证明训练、切分、推理接线和稳定性流程，不代表英雄素材收益。\n"
        "- 正式素材到达后必须用同一冻结配置重训并重新生成 metrics/manifest。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "complete", "artifact": str(artifact), "sha256": artifact_sha,
             "gate": gate},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
