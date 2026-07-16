#!/usr/bin/env python
"""Build and run the synthetic A1 narration robustness benchmark.

Audio is generated from deterministic, machine-known facts.  Cloud and local models
only extract those facts; their outputs are scored evidence, never accepted as truth.

Typical calibration:
  python scripts/a1_robustness.py init --run-dir local-data/a1/calibration \
    --base-cases 3 --conditions clean,noise20,speed090
  python scripts/a1_robustness.py cloud --run-dir local-data/a1/calibration \
    --max-new-tts 3 --max-new-extractions 9
  python scripts/a1_robustness.py score --run-dir local-data/a1/calibration \
    --report-dir results/acceptance/A1/calibration
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from backend.tools.a1_benchmark import (  # noqa: E402
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    aggregate_scores,
    build_plan,
    score_case,
)
from stepfun_api import (  # noqa: E402
    StepfunAPIError,
    chat_completion,
    synthesize_speech,
)

DEFAULT_CONDITIONS = "clean,noise20,noise10,speed090,codec32"
DEFAULT_VOICES = "cixingnansheng,linjiajiejie"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def load_plan(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "plan.json"
    if not path.exists():
        raise SystemExit(f"missing plan: {path}; run the init command first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"unsupported plan schema: {plan.get('schema_version')!r}")
    return plan


def retry_api(operation: Callable[[], Any], *, attempts: int = 2) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except StepfunAPIError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                print(f"API transient failure, retrying once: {exc}", file=sys.stderr)
                time.sleep(2.0)
    assert last_error is not None
    raise last_error


def ffmpeg(*arguments: str) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise SystemExit("ffmpeg is required for A1 audio normalization")
    process = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr[-2000:]}")


def canonical_wav(source: Path, output: Path, *, audio_filter: str | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.wav")
    command = ["-i", str(source), "-ac", "1", "-ar", "16000"]
    if audio_filter:
        command.extend(["-filter:a", audio_filter])
    command.extend(["-c:a", "pcm_s16le", str(temporary)])
    ffmpeg(*command)
    temporary.replace(output)


def add_noise_at_snr(source: Path, output: Path, *, snr_db: float, seed: int) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 16000:
            raise RuntimeError("noise input must be 16kHz mono PCM16 WAV")
        samples = array("h")
        samples.frombytes(reader.readframes(params.nframes))
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(float(value) * float(value) for value in samples) / max(1, len(samples)))
    target_noise_rms = rms / (10.0 ** (snr_db / 20.0))
    rng = random.Random(seed)
    noisy = array("h")
    for value in samples:
        mixed = round(value + rng.gauss(0.0, target_noise_rms))
        noisy.append(max(-32768, min(32767, mixed)))
    if sys.byteorder != "little":
        noisy.byteswap()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.wav")
    with wave.open(str(temporary), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(noisy.tobytes())
    temporary.replace(output)


def render_condition(source: Path, output: Path, condition: dict[str, Any], *, seed: int) -> None:
    if output.exists():
        return
    kind = condition["kind"]
    if kind == "clean":
        canonical_wav(source, output)
    elif kind == "speed":
        canonical_wav(source, output, audio_filter=f"atempo={float(condition['ratio']):.3f}")
    elif kind == "noise":
        with tempfile.TemporaryDirectory(prefix="a1-noise-") as directory:
            clean = Path(directory) / "clean.wav"
            canonical_wav(source, clean)
            add_noise_at_snr(clean, output, snr_db=float(condition["snr_db"]), seed=seed)
    elif kind == "codec":
        with tempfile.TemporaryDirectory(prefix="a1-codec-") as directory:
            clean = Path(directory) / "clean.wav"
            encoded = Path(directory) / "encoded.mp3"
            canonical_wav(source, clean)
            ffmpeg("-i", str(clean), "-b:a", str(condition["bitrate"]), str(encoded))
            canonical_wav(encoded, output)
    else:
        raise RuntimeError(f"unsupported audio condition: {kind!r}")


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def cmd_init(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if (run_dir / "plan.json").exists() and not args.force:
        raise SystemExit(f"plan already exists: {run_dir / 'plan.json'} (use --force to replace)")
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    voices = [value.strip() for value in args.voices.split(",") if value.strip()]
    plan = build_plan(
        seed=args.seed,
        base_cases=None if args.base_cases == 0 else args.base_cases,
        condition_ids=conditions,
        voices=voices,
        target_half_width=args.ci_half_width,
        minimum_per_condition=args.min_per_condition,
        maximum_observations=args.max_observations,
    )
    write_json(run_dir / "plan.json", plan)
    print(json.dumps({
        "run_dir": str(run_dir),
        "base_cases": plan["base_case_count"],
        "observations_per_backend": plan["observation_count_per_backend"],
        "conditions": conditions,
        "target_reachable": plan["stopping_rule"]["target_reachable_in_plan"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _source_paths(run_dir: Path, case_id: str) -> tuple[Path, Path]:
    source = run_dir / "audio" / "source" / f"{case_id}.wav"
    metadata = run_dir / "audio" / "source" / f"{case_id}.json"
    return source, metadata


def ensure_source_audio(
    run_dir: Path,
    case: dict[str, Any],
    *,
    tts_model: str,
) -> bool:
    source, metadata = _source_paths(run_dir, case["case_id"])
    if source.exists() and metadata.exists():
        return False
    audio = retry_api(lambda: synthesize_speech(
        text=case["narration"],
        model=tts_model,
        voice=case["voice"],
        instruction=case["tts_instruction"],
        response_format="wav",
    ))
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.with_name(source.stem + ".tmp.wav")
    temporary.write_bytes(audio)
    temporary.replace(source)
    write_json(metadata, {
        "case_id": case["case_id"],
        "created_at": utc_now(),
        "model": tts_model,
        "voice": case["voice"],
        "narration_sha256": sha256_bytes(case["narration"].encode()),
        "audio_sha256": sha256_file(source),
        "raw_bytes": source.stat().st_size,
    })
    return True


def prediction_path(run_dir: Path, backend: str, condition_id: str, case_id: str) -> Path:
    return run_dir / "predictions" / backend / condition_id / f"{case_id}.json"


def extract_cloud(
    *,
    audio: Path,
    output: Path,
    case_id: str,
    condition_id: str,
    model: str,
    temperature: float,
) -> None:
    encoded = base64.b64encode(audio.read_bytes()).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "解析这段旁白。"},
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
        ]},
    ]
    response = retry_api(lambda: chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    ))
    content = response["choices"][0]["message"]["content"]
    write_json(output, {
        "case_id": case_id,
        "condition_id": condition_id,
        "backend": "cloud",
        "model": model,
        "temperature": temperature,
        "created_at": utc_now(),
        "audio_sha256": sha256_file(audio),
        "raw_text": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        "usage": response.get("usage", {}),
    })


def cmd_cloud(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    plan = load_plan(run_dir)
    new_tts = 0
    new_extractions = 0
    for case in plan["cases"]:
        source, _ = _source_paths(run_dir, case["case_id"])
        if not source.exists():
            if new_tts >= args.max_new_tts:
                print("TTS safety cap reached; remaining cases left resumable", file=sys.stderr)
                break
            created = ensure_source_audio(run_dir, case, tts_model=args.tts_model)
            new_tts += int(created)
            print(f"tts {case['case_id']} voice={case['voice']}")
        for condition_id in plan["condition_ids"]:
            audio = run_dir / "audio" / condition_id / f"{case['case_id']}.wav"
            deterministic_seed = int(hashlib.sha256(
                f"{plan['seed']}:{case['case_id']}:{condition_id}".encode()
            ).hexdigest()[:16], 16)
            render_condition(source, audio, plan["conditions"][condition_id], seed=deterministic_seed)
            output = prediction_path(run_dir, "cloud", condition_id, case["case_id"])
            if output.exists():
                continue
            if new_extractions >= args.max_new_extractions:
                print("extraction safety cap reached; remaining cases left resumable", file=sys.stderr)
                metrics = score_run(run_dir, plan)
                print(json.dumps(_status_summary(metrics), ensure_ascii=False, sort_keys=True))
                return 0
            extract_cloud(
                audio=audio,
                output=output,
                case_id=case["case_id"],
                condition_id=condition_id,
                model=args.chat_model,
                temperature=args.temperature,
            )
            new_extractions += 1
            print(f"extract cloud {condition_id}/{case['case_id']}")
    metrics = score_run(run_dir, plan)
    print(json.dumps({
        **_status_summary(metrics),
        "new_tts_calls": new_tts,
        "new_extraction_calls": new_extractions,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def load_prediction(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("raw_text"), str):
        raise ValueError(f"prediction missing raw_text: {path}")
    return payload


def score_run(run_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in plan["cases"]}
    records: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "tts_calls": 0}
    source_dir = run_dir / "audio" / "source"
    usage["tts_calls"] = len(list(source_dir.glob("case-*.json"))) if source_dir.exists() else 0
    predictions_root = run_dir / "predictions"
    backends = sorted(path.name for path in predictions_root.iterdir() if path.is_dir()) if predictions_root.exists() else []
    for backend in backends:
        for condition_id in plan["condition_ids"]:
            for case_id, case in case_by_id.items():
                path = prediction_path(run_dir, backend, condition_id, case_id)
                if not path.exists():
                    continue
                payload = load_prediction(path)
                record = score_case(case, payload["raw_text"])
                record.update({
                    "backend": backend,
                    "condition_id": condition_id,
                    "model": payload.get("model"),
                    "temperature": payload.get("temperature"),
                    "audio_sha256": payload.get("audio_sha256"),
                })
                records.append(record)
                if backend == "cloud":
                    call_usage = payload.get("usage", {})
                    usage["prompt_tokens"] += int(call_usage.get("prompt_tokens") or 0)
                    usage["completion_tokens"] += int(call_usage.get("completion_tokens") or 0)
                    usage["calls"] += 1
    metrics = aggregate_scores(records, plan)
    metrics["generated_at"] = utc_now()
    metrics["usage"] = usage
    metrics["observed_predictions"] = len(records)
    metrics["expected_predictions_per_backend"] = plan["observation_count_per_backend"]
    write_json(run_dir / "metrics.json", metrics)
    results_path = run_dir / "case-results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return metrics


def _status_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "backends": {
            backend: {
                "cases": data["cases"],
                "slot_accuracy": data["slot_accuracy"]["rate"],
                "ci_half_width": data["slot_accuracy"]["half_width"],
                "stopping_reached": data["stopping"]["reached"],
            }
            for backend, data in metrics["backends"].items()
        },
        "usage": metrics["usage"],
    }


def git_revision(project_dir: Path = PROJ) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=project_dir, check=False
        ).returncode != 0
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        deployed = project_dir / "COMMIT"
        if deployed.exists():
            revision = deployed.read_text(encoding="utf-8").strip()
            return revision or "unknown"
        return "unknown"


def export_report(run_dir: Path, report_dir: Path, plan: dict[str, Any], metrics: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    plan_bytes = (run_dir / "plan.json").read_bytes()
    write_json(report_dir / "plan-summary.json", {
        "schema_version": plan["schema_version"],
        "seed": plan["seed"],
        "oracle": plan["oracle"],
        "privacy": plan["privacy"],
        "condition_ids": plan["condition_ids"],
        "voices": plan["voices"],
        "base_case_count": plan["base_case_count"],
        "observation_count_per_backend": plan["observation_count_per_backend"],
        "stopping_rule": plan["stopping_rule"],
        "style_coverage": sorted({case["style"] for case in plan["cases"]}),
        "item_count_coverage": sorted({case["item_count"] for case in plan["cases"]}),
        "completeness_coverage": sorted({case["completeness"] for case in plan["cases"]}),
        "plan_sha256": sha256_bytes(plan_bytes),
    })
    write_json(report_dir / "metrics.json", metrics)
    shutil.copyfile(run_dir / "case-results.jsonl", report_dir / "case-results.jsonl")
    truth_path = report_dir / "truth.jsonl"
    truth_path.write_text(
        "".join(json.dumps({
            "case_id": case["case_id"],
            "style": case["style"],
            "voice": case["voice"],
            "completeness": case["completeness"],
            "narration": case["narration"],
            "expected": case["expected"],
        }, ensure_ascii=False, sort_keys=True) + "\n" for case in plan["cases"]),
        encoding="utf-8",
    )
    write_json(report_dir / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "code_revision": git_revision(),
        "plan_sha256": sha256_bytes(plan_bytes),
        "audio_committed": False,
        "synthetic_audio_location": "local-data only; excluded from git and Spark deploy",
        "observed_predictions": metrics["observed_predictions"],
        "usage": metrics["usage"],
        "models": {
            backend: sorted({
                record.get("model")
                for record in _read_jsonl(run_dir / "case-results.jsonl")
                if record.get("backend") == backend and record.get("model")
            })
            for backend in metrics["backends"]
        },
    })


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cmd_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    plan = load_plan(run_dir)
    metrics = score_run(run_dir, plan)
    if args.report_dir:
        export_report(run_dir, Path(args.report_dir), plan, metrics)
    print(json.dumps(_status_summary(metrics), ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a deterministic plan")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--seed", type=int, default=20260716)
    init.add_argument("--base-cases", type=int, default=0, help="0 = derive from CI target")
    init.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    init.add_argument("--voices", default=DEFAULT_VOICES)
    init.add_argument("--ci-half-width", type=float, default=0.08)
    init.add_argument("--min-per-condition", type=int, default=100)
    init.add_argument("--max-observations", type=int, default=2000)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    cloud = sub.add_parser("cloud", help="synthesise, perturb, and extract with Step Plan")
    cloud.add_argument("--run-dir", required=True)
    cloud.add_argument("--tts-model", default="stepaudio-2.5-tts")
    cloud.add_argument("--chat-model", default="stepaudio-2.5-chat")
    cloud.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="formal benchmark default is deterministic greedy decoding",
    )
    cloud.add_argument("--max-new-tts", type=int, default=10)
    cloud.add_argument("--max-new-extractions", type=int, default=50)
    cloud.set_defaults(func=cmd_cloud)

    score = sub.add_parser("score", help="score every available backend prediction")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--report-dir", default=None)
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StepfunAPIError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
