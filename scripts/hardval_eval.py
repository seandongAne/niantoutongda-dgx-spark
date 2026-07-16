#!/usr/bin/env python
"""对机器可读 hardval GT/预测 JSON 输出固定三指标。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.detection_eval import (  # noqa: E402
    EvaluationInputError,
    evaluate_detection_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate hardval predictions with the frozen S2.5 metrics."
    )
    parser.add_argument("ground_truth", help="ground-truth JSON file")
    parser.add_argument("predictions", help="prediction JSON file")
    parser.add_argument("--output", help="write result JSON here instead of stdout")
    args = parser.parse_args(argv)

    try:
        evaluation = evaluate_detection_files(args.ground_truth, args.predictions)
        rendered = json.dumps(
            evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (EvaluationInputError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
