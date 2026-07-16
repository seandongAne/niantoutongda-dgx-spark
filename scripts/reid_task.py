#!/usr/bin/env python
"""S3 v5/v6 跨视频重识别入口。

产物写入 ``--out``；默认重复三次并比较规范化结果 hash，证明确定性。
没有 ``--annotations`` 真值时明确标记 baseline-only，不计算 G2 指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    # spark 上没有 .git;deploy.sh 会把当前 commit 写进 COMMIT 文件
    stamp = PROJ / "COMMIT"
    if stamp.exists():
        return stamp.read_text().strip() or "unknown"
    return "unknown"


def _run_payload(run) -> bytes:
    payload = {
        "entities": [entity.model_dump(mode="json") for entity in run.entities],
        "clarifications": [item.model_dump(mode="json") for item in run.clarifications],
        "candidates": run.candidates,
        "accepted_links": run.accepted_links,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--constraints")
    parser.add_argument("--attributes", help="S5 属性抽取产物 JSONL(可选)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    from backend.schemas.core import AuditEvent
    from backend.tools.reid.matcher import IdentityConstraints, run_reid
    from backend.tools.reid.model import ReIDConfig, Vocabulary, load_attribute_enrichment

    config_path = Path(args.config)
    vocab_path = Path(args.vocab)
    constraints_path = Path(args.constraints) if args.constraints else None
    attributes_path = Path(args.attributes) if args.attributes else None
    config = ReIDConfig.from_yaml(config_path)
    vocab = Vocabulary.from_json(vocab_path)
    constraints = IdentityConstraints.from_json(constraints_path)
    attributes = load_attribute_enrichment(attributes_path) if attributes_path else None

    runs = [
        run_reid(
            ingest_root=args.ingest_root,
            config=config,
            vocab=vocab,
            constraints=constraints,
            attributes=attributes,
        )
        for _ in range(args.repeat)
    ]
    run_hashes = [hashlib.sha256(_run_payload(run)).hexdigest() for run in runs]
    deterministic = len(set(run_hashes)) == 1
    first = runs[0]
    first.metrics.update(
        {
            "repeat_count": args.repeat,
            "run_hashes": run_hashes,
            "deterministic": deterministic,
        }
    )
    out = Path(args.out)
    first.write(out)

    manifest = {
        "schema_version": "1.0",
        "slice_id": "S3",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": _git_commit(),
        "ingest_root": str(args.ingest_root),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "vocab": {"path": str(vocab_path), "sha256": _sha256(vocab_path)},
        "constraints": {
            "path": str(constraints_path) if constraints_path else None,
            "sha256": _sha256(constraints_path),
        },
        "attributes": {
            "path": str(attributes_path) if attributes_path else None,
            "sha256": _sha256(attributes_path),
        },
        "ground_truth": None,
        "ground_truth_status": "MISSING_MACHINE_READABLE_ANNOTATIONS",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    event = AuditEvent(
        event_id=f"s3_{run_hashes[0][:12]}",
        event_type="CrossVideoReIDBaselineCompleted",
        actor="MEM",
        input_refs=[str(args.ingest_root), str(config_path), str(vocab_path)],
        output_refs=[str(out / "entities.jsonl"), str(out / "metrics.json")],
        config_version=config.version,
        created_at=manifest["created_at"],
    )
    (out / "audit-events.jsonl").write_text(event.model_dump_json() + "\n")
    (out / "failure-case.md").write_text(
        "# S3 baseline 未验收边界\n\n"
        "当前任务 A 尚无机器可读的 anchor→tracklet 真值，故本次只证明 S3 "
        "算法链与确定性，不能计算 Recall@1、完整合并数、高置信误合并或四组硬负结果。\n"
    )
    print(json.dumps(first.metrics, ensure_ascii=False, sort_keys=True))
    return 0 if deterministic else 2


if __name__ == "__main__":
    raise SystemExit(main())
