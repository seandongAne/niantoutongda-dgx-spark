#!/usr/bin/env python
"""新家区域 manifest 校验与归一 — new_1.mp4 → CandidateRegion 的人工登记通道。

校验 RegionManifest 契约(每区域必须带画面证据引用,near_power 同样吃
这条纪律),写归一化 regions.json + core Region JSONL。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.solver.region_adapter import (  # noqa: E402
    load_region_manifest,
    to_candidate_regions,
    to_core_regions,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    manifest = load_region_manifest(args.manifest)
    candidates = to_candidate_regions(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "regions.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "regions_core.jsonl").open("w", encoding="utf-8") as f:
        for region in to_core_regions(manifest):
            f.write(region.model_dump_json() + "\n")
    print(
        json.dumps(
            {
                "video_id": manifest.video_id,
                "regions": len(candidates),
                "near_power": sum(1 for r in candidates if r.near_power),
                "capacity_units_total": sum(r.capacity_units for r in candidates),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
