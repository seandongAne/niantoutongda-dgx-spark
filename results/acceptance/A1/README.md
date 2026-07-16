# A1 语音鲁棒性验收索引（2026-07-16）

## 当前裁决

- `CALIBRATION_VALID`：v3 的真值、扰动、同输入 A/B、判卷、运行时和用量证据完整。
- `COVERAGE_COMPLETE_FOR_CALIBRATION`：云端与本地均完成 18/18，覆盖率 100%。
- `STATISTICAL_STOP_NOT_REACHED`：每条件仅 6 条，冻结门要求 147 条；不得据此宣称
  云端或本地总体胜出。
- `FORMAL_EXPANSION_NOT_STARTED`：预注册的 147×5 正式批次尚未启动；启动需要对更大
  批次中的临时 key 暴露单独明确授权。

成功指标是覆盖完整性与 95% 区间稳定性。token/call 只作预算账本，不是越高越好的 KPI。

## 三轮证据

| run | 用途 | 结果 | 是否可比较 |
|---|---|---|---|
| `calibration-20260716-v1` | 首个协议探针 | 跑到 9 条时发现 reordered 模板先说去向再命名物品，TTS 后产生真值歧义，主动停止 | 否；只保留失败证据 |
| `calibration-20260716-v2` | 修正真值后的云端探索 | 18/18；temperature=0.2；字段 173/180=96.111% | 只用于验证 oracle/scorer，不与正式贪心 A/B 混算 |
| `calibration-20260716-v3-spark` | Spark 原地、同音频、双端贪心校准 | 云端与本地各 18/18；统一判卷完成 | 是；但样本量只够协议校准 |

## v3 核心指标

计划 SHA-256：`7d4a3ed32c2375a5af1ede090cc11634cc2516c6df9c2f7284c58fed582d8736`。
本地推理生成代码为 `5332060`；按 coverage KPI 重导出判卷的代码为 `3497a81`。

| 指标 | StepAudio 2.5 Chat（云） | Step-Audio 2 mini（Spark） |
|---|---:|---:|
| 覆盖 | 18/18（100%） | 18/18（100%） |
| Schema 合法 | 18/18（100%） | 17/18（94.444%） |
| 整条全对 | 9/18（50%） | 9/18（50%） |
| 物品召回 | 36/36（100%） | 36/36（100%） |
| 物品精度 | 36/36（100%） | 36/42（85.714%） |
| 字段准确率 | 157/180（87.222%） | 161/180（89.444%） |
| 95% Wilson 半宽（汇总） | 0.048881 | 0.045176 |
| 停止门 | 未达到（每条件 6/147） | 未达到（每条件 6/147） |

两端 18 组对应输入的音频 SHA-256 全部一致；每个 case 的 clean/noise20/speed090
三种哈希均互不相同。覆盖维度为 3 类话术、1–3 件物品、全字段/省略字段、双音色。

## 真实失败，不改 scorer 掩盖

- 云端物品识别稳定，但空间关系有损失：`source_location` 23/36、
  `target_location` 28/36；典型输出把“旧卧室书桌上”缩成“旧卧室书桌”。“上”仍按
  空间槽位信息计分，不作为无损同义词放宽。
- 本地 mini 的 owner/color 保真为 36/36，但 `pack_group` 28/36、
  `target_location` 26/36。在 case-0001 的 noise20/speed090 中，把旧书桌、新书桌、
  学习用品过抽取成 3 个额外物品；speed090 的额外项还产出 `attributes=null`，造成
  1 次 schema mismatch。
- 云/本地汇总字段准确率相差 2.22pp，但当前每条件只有 6 条，不能解释成总体优劣。

## 运行与用量

- 云端：6 次 TTS、18 次抽取、7542 prompt tokens、2813 completion tokens。
- 本地：模型加载 124.864549s；18 条均值 54.442838s；峰值 CUDA 分配
  17,392,368,128 bytes（约 16.20 GiB）。
- 工厂全程只在 Spark 原地保存 WAV 和预测池；通过 SSH 拉回的 acceptance 摘要为 64KB。
- key 只在云阶段进入工厂进程，进入本地阶段前已清除；平台 key 的撤销例外登记为
  批次完成后 6 天，由控制台执行。

机器可读证据以各 run 目录内的 `metrics.json`、`case-results.jsonl`、
`plan-summary.json`、`manifest.json` 为准；v3 另含 `factory-status.json`。
