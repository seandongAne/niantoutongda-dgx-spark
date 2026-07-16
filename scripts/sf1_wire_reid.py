#!/usr/bin/env python
"""把已验 hash 的 SF1 权重接入一份派生 ReID 配置，不改写冻结基线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.model import ReIDConfig
from backend.tools.sf1.projection import NumpyProjectionHead, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    raw = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("slice_id") != "SF1-L1":
        raise ValueError("manifest is not an SF1-L1 artifact")
    projection = manifest["projection"]
    artifact = args.artifact or Path(projection["path"])
    actual_sha = sha256_file(artifact)
    if actual_sha != projection["sha256"]:
        raise ValueError(
            f"projection sha256 mismatch: manifest={projection['sha256']} actual={actual_sha}"
        )
    head = NumpyProjectionHead.load(artifact, expected_sha256=actual_sha)
    if head.input_dim != int(raw["embedding_dim"]):
        raise ValueError(
            f"projection input_dim {head.input_dim} != ReID embedding_dim {raw['embedding_dim']}"
        )

    base_version = str(raw["version"])
    raw["version"] = f"{base_version}+sf1-{actual_sha[:8]}"
    raw["frozen"] = False
    raw["derived_from"] = {
        "base_config": str(args.base),
        "base_sha256": sha256_file(args.base),
        "sf1_manifest": str(args.manifest),
        "sf1_manifest_sha256": sha256_file(args.manifest),
    }
    raw["projection"] = {
        "enabled": True,
        "artifact": str(artifact),
        "sha256": actual_sha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    # 反读是接线门：字段拼错或 artifact 缺失时不产出“看起来可用”的配置。
    ReIDConfig.from_yaml(args.out)
    print(
        json.dumps(
            {"out": str(args.out), "version": raw["version"], "projection_sha256": actual_sha},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
