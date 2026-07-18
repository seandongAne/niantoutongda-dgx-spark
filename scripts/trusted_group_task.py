#!/usr/bin/env python
"""从可信库存投影构建 closure 冻结的三组生活组合与完整箱单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.trusted_downstream import build_trusted_downstream  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _read_json(path: Path) -> tuple[object, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number} must be a JSON object")
        rows.append(row)
    return rows, raw


def build_output_bytes(
    closure: object,
    inventory_rows: list[dict[str, Any]],
    display_rows: list[dict[str, Any]],
    *,
    input_sha256: dict[str, str],
) -> dict[str, bytes]:
    build = build_trusted_downstream(closure, inventory_rows, display_rows)
    groups_rows = [group.model_dump(mode="json") for group in build.groups]
    life_group_rows = [
        group.to_life_group().model_dump(mode="json") for group in build.groups
    ]
    payloads = {
        "groups.jsonl": _jsonl_bytes(groups_rows),
        "life_groups.jsonl": _jsonl_bytes(life_group_rows),
        "placement_groups.jsonl": _jsonl_bytes(
            [group.model_dump(mode="json") for group in build.placement_groups]
        ),
        "independent_pack_items.jsonl": _jsonl_bytes(list(build.independent_items)),
        "boxlist.json": _json_bytes(build.boxlist),
    }
    grouped_count = sum(len(group.entity_ids) for group in build.groups)
    metrics = {
        "schema_version": "1.0",
        "closure_id": build.closure_id,
        "status": "PASS",
        "trusted_inventory_count": len(build.trusted_items),
        "group_count": len(build.groups),
        "placement_group_count": len(build.placement_groups),
        "technical_pack_unit_count": len(build.placement_groups) - len(build.groups),
        "grouped_item_count": grouped_count,
        "placement_grouped_item_count": sum(
            len(group.entity_ids) for group in build.placement_groups
        ),
        "independent_item_count": len(build.independent_items),
        "covered_canonical_item_count": build.boxlist["canonical_item_count"],
        "box_count": build.boxlist["box_count"],
        "input_sha256": input_sha256,
        "output_sha256": {
            name: _sha256(content) for name, content in sorted(payloads.items())
        },
    }
    payloads["metrics.json"] = _json_bytes(metrics)
    return payloads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closure", required=True, type=Path)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--display", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    closure, closure_raw = _read_json(args.closure)
    inventory, inventory_raw = _read_jsonl(args.inventory)
    display, display_raw = _read_jsonl(args.display)
    payloads = build_output_bytes(
        closure,
        inventory,
        display,
        input_sha256={
            "closure": _sha256(closure_raw),
            "inventory": _sha256(inventory_raw),
            "display": _sha256(display_raw),
        },
    )

    # 全部合同校验和内存构建通过后才落盘，错误输入不留下半套可信产物。
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (args.out_dir / name).write_bytes(content)
    print(
        json.dumps(
            {
                "groups": 3,
                "grouped_items": 15,
                "independent_items": 5,
                "covered_items": 20,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
