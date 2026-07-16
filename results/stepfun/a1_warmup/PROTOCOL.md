# A1 旁白解析协议 v1(云上预热冻结稿,2026-07-16)

> 用途:拍摄旁白(语音)→ 物品结构化标签。云上(stepaudio-2.5-chat)调通后,
> 权重就绪时平移到本地 Step-Audio 2 mini,协议不变。
> 预热素材 = stepaudio-2.5-tts 合成语音,不涉家庭音频出境。

## System prompt(冻结)

```
你是搬家助手的旁白解析器。输入是用户拍摄房间时的口述旁白语音。抽取旁白中提到的每一件物品,
输出 JSON 数组,每项格式: {"label_zh": 中文名, "label_en": 英文检测短语(1-3词),
"owner": 所属人或null, "source_location": 当前位置或null, "target_location": 搬运去向或null,
"pack_group": 同包分组要求或null, "attributes": {"color": 颜色或null}}。
只输出 JSON 数组,不要任何解释。
```

User turn 固定为 "解析这段旁白。" + input_audio(wav)。temperature=0.2。

## 回环验证(2026-07-16)

- TTS:`scripts/stepfun_api.py tts --model stepaudio-2.5-tts --voice linjiajiejie`
  → `test_narration.wav`(852KB,三物品测试旁白)。
- 抽取:stepaudio-2.5-chat,输出 `extraction_output.json` — 3/3 物品、
  owner/source/target/pack_group/color 全部正确落位,无多余文本。
- token:tts 一次 + chat prompt=389 completion=211。

## 与主链的对接点

- `label_en` 直接可作 GDINO 补充检测词(A1→S2.5-3 通道)。
- `owner/pack_group/target_location` 进 GROUP/EXEC 的任务分组证据,
  云输出只作候选,入契约对象前须人工确认(playbook 红线 3)。

## 待办(本地化时)

- 本地 Step-Audio 2 mini 上复测同一 system prompt;若 JSON 稳定性下降,
  加 few-shot 一例。
- 真实拍摄旁白噪声(脚步/风扇)下的鲁棒性未测——用赛程内自录素材补,
  不用家庭历史音频。
