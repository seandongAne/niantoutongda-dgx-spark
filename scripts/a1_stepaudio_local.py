#!/usr/bin/env python
"""Run A1 benchmark audio through Spark-local Step-Audio 2 mini once per clip.

This script has no cloud credentials and is intended for ``~/envs/stepaudio`` on Spark.
Model weights stay in ``~/models``.  The audio preprocessing and chat-template mechanics
follow the official Step-Audio2 implementation at commit
76e272b56c3917a8d7188f18bbb5a65dfc8a0845 (Apache-2.0), reduced to text-output inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import librosa  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

from backend.tools.a1_benchmark import SCHEMA_VERSION, SYSTEM_PROMPT  # noqa: E402


UPSTREAM_REVISION = "76e272b56c3917a8d7188f18bbb5a65dfc8a0845"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_audio(path: Path, target_rate: int = 16_000) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_rate)(waveform)
    return waveform[0]


def mel_filters() -> torch.Tensor:
    return torch.from_numpy(librosa.filters.mel(sr=16_000, n_fft=400, n_mels=128))


def log_mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    audio = F.pad(audio, (0, 479))
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    spectrum = mel_filters().to(audio.device) @ magnitudes
    log_spec = torch.clamp(spectrum, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def audio_token_count(feature_length: int) -> int:
    encoder_output = (feature_length - 2 + 1) // 2 // 2
    return (encoder_output + 2 - 3) // 2 + 1


def pad_mels(mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([mel.size(1) - 2 for mel in mels], dtype=torch.int32)
    padded = pad_sequence([mel.t() for mel in mels], batch_first=True, padding_value=0)
    return padded.transpose(1, 2), lengths


class LocalStepAudio2:
    def __init__(self, model_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).cuda().eval()
        self.tokenizer.eos_token = "<|EOT|>"
        self.model.config.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.eos_token_id = self.model.config.eos_token_id

    def _template(self, audio_path: Path) -> tuple[list[str], list[torch.Tensor]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "human", "content": [
                {"type": "text", "text": "解析这段旁白。"},
                {"type": "audio", "audio": audio_path},
            ]},
            {"role": "assistant", "content": None},
        ]
        fragments: list[str] = []
        mels: list[torch.Tensor] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                fragments.append(f"<|BOT|>{role}\n{content}<|EOT|>")
            elif isinstance(content, list):
                fragments.append(f"<|BOT|>{role}\n")
                for item in content:
                    if item["type"] == "text":
                        fragments.append(item["text"])
                        continue
                    waveform = load_audio(Path(item["audio"]))
                    for start in range(0, waveform.shape[0], 16_000 * 25):
                        mel = log_mel_spectrogram(waveform[start:start + 16_000 * 25])
                        mels.append(mel)
                        fragments.append(
                            "<audio_start>"
                            + "<audio_patch>" * audio_token_count(mel.shape[1])
                            + "<audio_end>"
                        )
                fragments.append("<|EOT|>")
            elif content is None:
                fragments.append(f"<|BOT|>{role}\n")
            else:
                raise TypeError(f"unsupported message content: {type(content)}")
        return fragments, mels

    @torch.inference_mode()
    def extract(self, audio_path: Path) -> tuple[str, int]:
        fragments, mels = self._template(audio_path)
        prompt_parts = [
            self.tokenizer(text=fragment, return_tensors="pt", padding=True)["input_ids"]
            for fragment in fragments
        ]
        prompt_ids = torch.cat(prompt_parts, dim=-1).cuda()
        padded_mels, mel_lengths = pad_mels(mels)
        config = GenerationConfig(
            max_new_tokens=800,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
            do_sample=False,
        )
        outputs = self.model.generate(
            input_ids=prompt_ids,
            wavs=padded_mels.cuda(),
            wav_lens=mel_lengths.cuda(),
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=config,
            tokenizer=self.tokenizer,
        )
        generated = outputs[0, prompt_ids.shape[-1]:].tolist()
        if generated and generated[-1] == self.eos_token_id:
            generated = generated[:-1]
        text_tokens = [token for token in generated if token < 151_688]
        return self.tokenizer.decode(text_tokens), len(generated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / "models" / "stepfun-ai__Step-Audio-2-mini",
    )
    parser.add_argument("--backend", default="local")
    parser.add_argument("--max-new", type=int, default=50)
    args = parser.parse_args(argv)

    plan_path = args.run_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != SCHEMA_VERSION:
        parser.error(f"unsupported plan schema: {plan.get('schema_version')!r}")
    tasks: list[tuple[dict[str, Any], str, Path, Path]] = []
    for case in plan["cases"]:
        for condition_id in plan["condition_ids"]:
            audio = args.run_dir / "audio" / condition_id / f"{case['case_id']}.wav"
            output = args.run_dir / "predictions" / args.backend / condition_id / f"{case['case_id']}.json"
            if not audio.exists():
                parser.error(f"missing synthetic audio: {audio}")
            if not output.exists():
                tasks.append((case, condition_id, audio, output))
    tasks = tasks[:args.max_new]
    if not tasks:
        print("A1_LOCAL_NOTHING_TO_DO")
        return 0

    before = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    model = LocalStepAudio2(args.model_path)
    load_seconds = time.perf_counter() - load_started
    print(f"model_loaded seconds={load_seconds:.2f} free_before={before[0]}", flush=True)

    timings: list[float] = []
    for index, (case, condition_id, audio, output) in enumerate(tasks, start=1):
        started = time.perf_counter()
        text, generated_tokens = model.extract(audio)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        write_json(output, {
            "case_id": case["case_id"],
            "condition_id": condition_id,
            "backend": args.backend,
            "model": "stepfun-ai/Step-Audio-2-mini",
            "model_path": str(args.model_path),
            "upstream_revision": UPSTREAM_REVISION,
            "created_at": utc_now(),
            "audio_sha256": file_sha256(audio),
            "raw_text": text,
            "generated_tokens": generated_tokens,
            "latency_seconds": round(elapsed, 6),
        })
        print(
            f"[{index}/{len(tasks)}] {condition_id}/{case['case_id']} "
            f"seconds={elapsed:.2f} tokens={generated_tokens}",
            flush=True,
        )

    write_json(args.run_dir / "predictions" / args.backend / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model": "stepfun-ai/Step-Audio-2-mini",
        "model_path": str(args.model_path),
        "upstream_revision": UPSTREAM_REVISION,
        "new_predictions": len(tasks),
        "model_load_seconds": round(load_seconds, 6),
        "mean_inference_seconds": round(sum(timings) / len(timings), 6),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    })
    print("A1_LOCAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
