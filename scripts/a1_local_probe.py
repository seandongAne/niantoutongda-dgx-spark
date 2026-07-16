#!/usr/bin/env python
"""A1 本地化复测 — Step-Audio-2-mini 跑冻结协议 v1(旁白→物品结构化)。

协议出处 results/stepfun/a1_warmup/PROTOCOL.md:system prompt 冻结原文、
user = "解析这段旁白。" + 旁白音频、temperature=0.2。判卷 = 与云上
stepaudio-2.5-chat 参考输出(--reference)逐件比对 label_zh 与五要素槽位。

报文构造与音频编码遵循官方 stepfun-ai/Step-Audio2 仓库 stepaudio2.py /
utils.py(Apache-2.0):<|BOT|>role\\n…<|EOT|> 模板、25s 分块 128-mel、
<audio_start><audio_patch>*N<audio_end> 占位、generate(wavs, wav_lens)。
在 ~/envs/stepaudio(transformers==4.49.0)内运行,勿用主 venv。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

SYSTEM_PROMPT = (
    "你是搬家助手的旁白解析器。输入是用户拍摄房间时的口述旁白语音。抽取旁白中提到的每一件物品,\n"
    '输出 JSON 数组,每项格式: {"label_zh": 中文名, "label_en": 英文检测短语(1-3词),\n'
    '"owner": 所属人或null, "source_location": 当前位置或null, "target_location": 搬运去向或null,\n'
    '"pack_group": 同包分组要求或null, "attributes": {"color": 颜色或null}}。\n'
    "只输出 JSON 数组,不要任何解释。"
)
USER_TEXT = "解析这段旁白。"
SLOT_KEYS = ("owner", "source_location", "target_location", "pack_group")

# few-shot 一例(协议 v1 待办预案):示范 pack_group 与 target_location 的
# 区分、attributes 恒为对象、旁白没说的槽位置 null。system prompt 不动。
FEW_SHOT_USER = (
    "解析这段旁白。旁白:绿色台灯,哥哥的,现在在客厅角落,"
    "搬过去放书房桌上,和路由器打包在一起。"
)
FEW_SHOT_ASSISTANT = json.dumps(
    [
        {
            "label_zh": "绿色台灯",
            "label_en": "green desk lamp",
            "owner": "哥哥",
            "source_location": "客厅角落",
            "target_location": "书房桌上",
            "pack_group": "路由器",
            "attributes": {"color": "绿色"},
        }
    ],
    ensure_ascii=False,
)


# ---- 以下三个辅助函数照抄官方 utils.py(Apache-2.0);模型仓的
# modeling 文件用相对导入,无法作为顶层模块引用,故内联。 ----


def load_audio(file_path: str, target_rate: int = 16000) -> torch.Tensor:
    # 新版 torchaudio.load 依赖 torchcodec(env 未装);soundfile 解码 +
    # torchaudio 纯张量重采样,行为与官方 load_audio 等价(单声道、16k)。
    import soundfile as sf
    import torchaudio

    data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)
    return waveform[0]


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = 128, padding: int = 479):
    import librosa
    import torch.nn.functional as F

    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def compute_token_num(max_feature_len: int) -> int:
    encoder_output_dim = (max_feature_len - 2 + 1) // 2 // 2
    return (encoder_output_dim + 2 * 1 - 3) // 2 + 1


def padding_mels(mels: list[torch.Tensor]):
    """官方 utils.padding_mels 等价实现:(128,T) 列表 → (B,128,Tmax) 与长度。"""
    lengths = torch.tensor([m.size(1) - 2 for m in mels], dtype=torch.int32)
    feats = [m.t() for m in mels]
    max_len = max(f.size(0) for f in feats)
    batch = torch.zeros(len(feats), max_len, feats[0].size(1), dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        batch[i, : f.size(0)] = f
    return batch.transpose(1, 2), lengths


def build_prompt(tokenizer, model_dir: str, audio_path: str, few_shot: bool = False):
    audio = load_audio(audio_path)
    mels, audio_segments = [], []
    for i in range(0, audio.shape[0], 16000 * 25):
        mel = log_mel_spectrogram(audio[i : i + 16000 * 25], n_mels=128, padding=479)
        mels.append(mel)
        n_tokens = compute_token_num(mel.shape[1])
        audio_segments.append(f"<audio_start>{'<audio_patch>' * n_tokens}<audio_end>")
    example = (
        f"<|BOT|>human\n{FEW_SHOT_USER}<|EOT|>"
        f"<|BOT|>assistant\n{FEW_SHOT_ASSISTANT}<|EOT|>"
    ) if few_shot else ""
    text = (
        f"<|BOT|>system\n{SYSTEM_PROMPT}<|EOT|>"
        f"{example}"
        f"<|BOT|>human\n{USER_TEXT}{''.join(audio_segments)}<|EOT|>"
        f"<|BOT|>assistant\n"
    )
    ids = tokenizer(text=text, return_tensors="pt")["input_ids"]
    wavs, wav_lens = padding_mels(mels)
    return ids, wavs, wav_lens, audio.shape[0] / 16000


def parse_json_array(text: str):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"输出中未找到 JSON 数组: {text[:200]!r}")
    return json.loads(m.group(0))


def judge(items: list[dict], reference: list[dict]) -> dict:
    ref_by_label = {r["label_zh"]: r for r in reference}
    per_item, ok = [], 0
    for r_label, ref in ref_by_label.items():
        got = next((i for i in items if i.get("label_zh") == r_label), None)
        if got is None:
            per_item.append({"label_zh": r_label, "verdict": "MISSING"})
            continue
        diffs = [
            k for k in SLOT_KEYS if got.get(k) != ref.get(k)
        ]
        if (got.get("attributes") or {}).get("color") != (ref.get("attributes") or {}).get("color"):
            diffs.append("attributes.color")
        per_item.append(
            {"label_zh": r_label, "verdict": "PASS" if not diffs else "DIFF",
             "diffs": diffs}
        )
        ok += not diffs
    return {
        "items_expected": len(reference),
        "items_extracted": len(items),
        "items_pass": ok,
        "per_item": per_item,
        "extra_items": [i.get("label_zh") for i in items
                        if i.get("label_zh") not in ref_by_label],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    t_load = time.time() - t0

    ids, wavs, wav_lens, audio_seconds = build_prompt(
        tokenizer, args.model, args.audio, few_shot=args.few_shot
    )
    t1 = time.time()
    outputs = model.generate(
        input_ids=ids.cuda(),
        attention_mask=torch.ones_like(ids).cuda(),
        wavs=wavs.cuda(),
        wav_lens=wav_lens.cuda(),
        generation_config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature or None,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ),
        tokenizer=tokenizer,
    )
    t_gen = time.time() - t1
    out_ids = outputs[0, ids.shape[-1]:].tolist()
    text = tokenizer.decode(
        [i for i in out_ids if i < 151688], skip_special_tokens=True
    )

    result = {
        "model": args.model,
        "audio": args.audio,
        "audio_seconds": round(audio_seconds, 1),
        "temperature": args.temperature,
        "few_shot": args.few_shot,
        "load_s": round(t_load, 1),
        "generate_s": round(t_gen, 1),
        "new_tokens": len(out_ids),
        "raw_text": text,
    }
    try:
        items = parse_json_array(text)
        result["items"] = items
        result["json_ok"] = True
        if args.reference:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            result["judgement"] = judge(items, reference)
    except (ValueError, json.JSONDecodeError) as exc:
        result["json_ok"] = False
        result["error"] = str(exc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "a1_local_result_fewshot.json" if args.few_shot else "a1_local_result.json"
    (args.out_dir / out_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {k: result.get(k) for k in
         ("json_ok", "load_s", "generate_s", "new_tokens")}
        | {"judgement": result.get("judgement", {}).get("items_pass")},
        ensure_ascii=False,
    ))
    return 0 if result.get("json_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
