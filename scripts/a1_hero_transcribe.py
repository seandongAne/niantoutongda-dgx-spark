#!/usr/bin/env python
"""英雄旁白本地誊写 — 静音切分 + Step-Audio 2 mini 逐段零-shot 贪心转写。

主链口径(PROTOCOL.md 本地化裁决):mini 的可靠边界是"听得准、认得出",
主链走"本地誊写 + 确定性 narration 解析器"。本脚本只做誊写:
narration.wav → 按静音切分为逐件口述段 → 每段贪心转写一行 →
transcript_draft.txt(供人工校对后落 fixtures/hero_s1/transcript.txt)。

音频报文构造(mel/token 计数/padding/生成参数)逐字复用
scripts/a1_stepaudio_local.py 的实现——该构造承载三连败教训
(相对导入/torchcodec/padding_mels 转置),勿改。
运行环境:spark ~/envs/stepaudio,权重在 ~/models,无云调用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

SCHEMA_VERSION = "a1-hero-transcribe-v1"

SYSTEM_PROMPT = (
    "你是搬家助手的旁白誊写员。输入是用户拍摄房间物品时的一段中文口述。"
    "请逐字转写这段话,保留原始口语表述,只输出转写文本,不要任何解释、翻译或补充。"
)
USER_TEXT = "誊写这段旁白。"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- 以下音频/报文机制复制自 a1_stepaudio_local.py(勿改)----

def load_audio(path: Path, target_rate: int = 16_000) -> torch.Tensor:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
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
                {"type": "text", "text": USER_TEXT},
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
    def transcribe(self, audio_path: Path, max_new: int) -> tuple[str, int]:
        fragments, mels = self._template(audio_path)
        prompt_parts = [
            self.tokenizer(text=fragment, return_tensors="pt", padding=True)["input_ids"]
            for fragment in fragments
        ]
        prompt_ids = torch.cat(prompt_parts, dim=-1).cuda()
        padded_mels, mel_lengths = pad_mels(mels)
        config = GenerationConfig(
            max_new_tokens=max_new,
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

# ---- 复制段结束 ----


def split_segments(
    audio: np.ndarray, rate: int, top_db: float, min_dur: float, pad: float
) -> list[tuple[int, int]]:
    """静音切分;短段并入前段,段边界各留 pad 秒。"""
    raw = librosa.effects.split(audio, top_db=top_db)
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and (end - start) / rate < min_dur:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    pad_n = int(pad * rate)
    return [
        (max(0, s - pad_n), min(len(audio), e + pad_n)) for s, e in merged
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / "models" / "stepfun-ai__Step-Audio-2-mini",
    )
    ap.add_argument("--top-db", type=float, default=40.0)
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    out_dir = args.out_dir
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    spans = split_segments(mono, rate, args.top_db, args.min_dur, args.pad)
    print(f"[split] {len(spans)} segments (top_db={args.top_db})", flush=True)

    model = LocalStepAudio2(args.model_path)
    started = time.time()
    lines: list[str] = []
    records: list[dict] = []
    for idx, (s, e) in enumerate(spans):
        seg_path = seg_dir / f"seg_{idx:03d}.wav"
        sf.write(seg_path, mono[s:e], rate)
        text, n_tokens = model.transcribe(seg_path, args.max_new)
        text = " ".join(text.split())
        lines.append(text)
        records.append({
            "index": idx,
            "start_s": round(s / rate, 2),
            "end_s": round(e / rate, 2),
            "text": text,
            "generated_tokens": n_tokens,
        })
        print(f"[{idx + 1}/{len(spans)}] {s / rate:7.2f}-{e / rate:7.2f}s {text}", flush=True)

    (out_dir / "segments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (out_dir / "transcript_draft.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "audio": str(args.audio),
        "audio_sha256": file_sha256(args.audio),
        "model_path": str(args.model_path),
        "decoding": {"do_sample": False, "temperature": None, "shots": 0},
        "split": {"top_db": args.top_db, "min_dur": args.min_dur, "pad": args.pad},
        "segments": len(spans),
        "elapsed_s": round(time.time() - started, 1),
        "created_at": utc_now(),
        "note": "誊写草稿,须人工校对后写入 fixtures/hero_s1/transcript.txt",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {len(spans)} lines -> {out_dir}/transcript_draft.txt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
