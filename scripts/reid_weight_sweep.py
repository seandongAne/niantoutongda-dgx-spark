#!/usr/bin/env python
"""S3 属性权重 A/B:attr 权重从 instance 让渡,阈值与其余字段锁定基准配置。

对照组 = attr 0.00(基准原样);处理组网格默认 0.10/0.15/0.20。
产物交给 anchor_gt_eval 判卷;本脚本不读 GT、不宣布优胜。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.matcher import IdentityConstraints, run_reid  # noqa: E402
from backend.tools.reid.model import ReIDConfig, Vocabulary, load_attribute_enrichment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", required=True, help="基准配置(阈值等不动)")
    parser.add_argument("--attributes", required=True, help="S5 属性抽取产物 JSONL")
    parser.add_argument("--attr-weights", default="0.00,0.10,0.15,0.20")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = ReIDConfig.from_yaml(args.config)
    vocab = Vocabulary.from_json(args.vocab)
    attributes = load_attribute_enrichment(args.attributes)
    out = Path(args.out)
    summary = []
    for attr_weight in (float(x) for x in args.attr_weights.split(",")):
        instance = base.weights.instance + base.weights.attribute - attr_weight
        if instance <= 0:
            raise ValueError(f"attr weight {attr_weight} leaves non-positive instance weight")
        version = f"{base.version}-wattr{attr_weight:.2f}"
        config = replace(
            base,
            version=version,
            weights=replace(base.weights, instance=instance, attribute=attr_weight),
        )
        config.validate()
        run = run_reid(
            ingest_root=args.ingest_root,
            config=config,
            vocab=vocab,
            constraints=IdentityConstraints(),
            attributes=attributes,
        )
        combo_dir = out / f"wattr{attr_weight:.2f}"
        run.write(combo_dir)
        summary.append(
            {
                "attr_weight": attr_weight,
                "instance_weight": round(instance, 4),
                "automatic_link_count": run.metrics["automatic_link_count"],
                "clarification_count": run.metrics["clarification_count"],
                "matched_entity_count": run.metrics["matched_entity_count"],
                "attribute_enriched_tracklet_count": run.metrics["attribute_enriched_tracklet_count"],
            }
        )
        print(json.dumps(summary[-1]))
    (out / "sweep-summary.json").write_text(
        json.dumps({"base_config": base.version, "attributes": args.attributes, "grid": summary},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
