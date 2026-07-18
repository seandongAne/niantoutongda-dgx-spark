#!/usr/bin/env python
"""独立验收 worker：消费 EXEC requests，仅产出 MEM 或 SPACE 结果。

两个 role 进程读取同一组不可变 request、任务卡与验收清单，但不读取彼此
的结果。输出是可直接并入正式 trace 的独立 fragment。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import VerificationCheckRequest  # noqa: E402
from backend.schemas.hero_bundle import AcceptanceManifest, TaskCard  # noqa: E402
from backend.tools.trace import load_trace, write_fragment  # noqa: E402
from backend.tools.verification.acceptance import (  # noqa: E402
    build_compliance_result,
    build_presence_result,
    validate_verification_request,
)


def _load_cards(path: Path) -> dict[str, TaskCard]:
    cards = [
        TaskCard.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {card.card_id: card for card in cards}
    if len(by_id) != len(cards):
        raise ValueError(f"{path}: duplicate task card_id")
    return by_id


def _load_requests(path: Path) -> list[VerificationCheckRequest]:
    rows = load_trace(path)
    requests = [row for row in rows if isinstance(row, VerificationCheckRequest)]
    if len(requests) != len(rows):
        raise ValueError(f"{path}: request fragment contains non-request messages")
    if not requests:
        raise ValueError(f"{path}: request fragment is empty")
    message_ids = [request.message_id for request in requests]
    task_ids = [request.task_id for request in requests]
    if len(message_ids) != len(set(message_ids)):
        raise ValueError(f"{path}: duplicate request message_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path}: duplicate request task_id")
    return requests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True, choices=("MEM", "SPACE"))
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--cards", required=True, type=Path)
    ap.add_argument(
        "--photos",
        "--acceptance",
        dest="photos",
        required=True,
        type=Path,
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    requests = _load_requests(args.requests)
    cards = _load_cards(args.cards)
    acceptance = AcceptanceManifest.model_validate_json(
        args.photos.read_text(encoding="utf-8")
    )

    build_result = (
        build_presence_result if args.role == "MEM" else build_compliance_result
    )
    results = []
    for request in requests:
        if not acceptance.includes_card(request.task_id):
            raise ValueError(
                f"{request.message_id}: task card is outside acceptance scope"
            )
        card = cards.get(request.task_id)
        if card is None:
            raise ValueError(f"{request.message_id}: task card {request.task_id!r} missing")
        validate_verification_request(card, request)
        results.append(build_result(request, acceptance))

    write_fragment(args.out, results)
    print(
        json.dumps(
            {"role": args.role, "requests": len(requests), "results": len(results)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
