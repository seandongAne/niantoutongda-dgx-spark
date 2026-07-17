# DAY-05 · 2026-07-17

> 赛程日：D5
>
> 当日主责：A1 formal 语音鲁棒性验收收口
>
> 状态：进行中（A1 formal 已完成；其余 D5 工作继续）

## 今日目标

- 在不重跑 StepFun、TTS 或 11 小时本地推理的前提下，修复 formal 最终评分的
  `attributes=null` 空值崩溃，完成预注册 147×5 双端 A/B 验收。
- 拉回并固化小体积机器可读证据，明确覆盖门、统计门、产品边界、失败恢复路径和
  key 撤销时间，不把额度消耗或单一汇总数字替代验收目标。

## 关键证据或截图

- 完成结果：
  - formal 原 worker 已完整生成 147 次 TTS、735 份 StepAudio 2.5 Chat 云预测和
    735 份 Step-Audio 2 mini 本地预测；本地运行均值 53.326959s、加载
    149.029278s、峰值 CUDA 17,392,470,528 bytes。
  - 原 worker 于 `2026-07-17T04:57:52Z` 在 score 阶段失败，真实原因是模型输出
    `attributes=null`，而 scorer 在 schema mismatch 路径仍假定该字段为对象并调用
    `.get()`。预测、真值和音频均未损坏。
  - scorer 修复保持冻结口径：`attributes=null` 继续记为 `schema_mismatch`，颜色槽位
    按缺失值计分；只消除崩溃，不放宽 schema、不改预测、不改真值、不改分母。
  - score-only 于 `2026-07-17T05:56:32Z` 成功导出 1470 行 case results；云/本地均
    735/735，五个条件各 147/147，`coverage_progress.complete=true`、
    `stopping.reached=true`、`additional_base_cases_needed=0`。
  - 正式字段准确率为云端 6692/7350=91.0476%、本地 6527/7350=88.8027%；本地
    物品召回 1462/1470=99.4558%，但 schema 合法率 704/735=95.7823%、物品精度
    1462/1629=89.7483%、整条全对 320/735=43.5374%，因此主链继续采用“本地誊写 +
    确定性解析”，mini 结构化输出只作能力与对照证据。
- Commit：`2f0de29`（null attributes 严格计错但不中断评分）。
- 验收证据：
  - `results/acceptance/A1/formal-20260716-v1-spark/manifest.json`
  - `results/acceptance/A1/formal-20260716-v1-spark/metrics.json`
  - `results/acceptance/A1/formal-20260716-v1-spark/case-results.jsonl`
  - `results/acceptance/A1/formal-20260716-v1-spark/truth.jsonl`
  - `results/acceptance/A1/formal-20260716-v1-spark/plan-summary.json`
- 日志 / metrics / 截图：
  - manifest：`code_revision=2f0de29a69882d4ecdbc9aeb442e581f3ee18970`、
    `observed_predictions=1470`、计划 SHA-256
    `a0a44b752eeae290c97b409be7700c0b9d1d5b69c2f417d0b6585a2560a2e7d0`。
  - 云端用量：147 次 TTS、735 次抽取、306585 prompt + 114918 completion tokens；
    云阶段于 `2026-07-16T18:02:05Z` 结束并释放 worker 内 key。
  - 本地验证：A1 专项 `11 passed`，全仓 `149 passed in 5.67s`；score-only 真机命令
    退出码 0，结果已通过 `scripts/pull_results.sh` 拉回本地。
- 未验证或降级边界：
  - 停止门证明两端覆盖和各自估计精度达到预注册要求，不等同于完成成对优劣显著性
    检验；当前不追加样本，也不把 2.2449pp 字段准确率差异扩写成产品总体胜负。
  - 原 `factory-status.json` 保留 failed 历史，不覆盖伪装为一次成功；最终验收由
    score-only 生成且带 `2f0de29` provenance 的五个 acceptance 文件承载。
  - 按延迟 6 天安排，StepFun 控制台 key 撤销时点登记为
    `2026-07-23T05:56:32Z`，仍需届时执行控制台动作。

## 失败与教训

- 协议校验正确识别 schema mismatch，并不保证评分路径能够处理非法字段类型。评测器
  必须把“严格判错”和“继续完成全量评分”同时作为契约；模型坏输出应成为失败样本，
  不能让整个统计批次失去报告。
- formal 的生成阶段和验收阶段必须分别留 provenance。原 worker 失败事实、恢复 commit
  与 score-only 报告同时保留，避免用最终成功文件抹去真实故障。
- scorer 小修部署时检测到另一并行会话正通过 SSH rsync 传输约 1.85GB hero 视频，
  ControlMaster 被占用并使 KB 级代码部署排队。该路径违反“大文件不得经 SSH/rsync
  传输”的项目纪律；本任务未中断并行进程，改用独立 SSH 连接完成小代码部署。后续复验
  确认上传进程已退出，Spark 只落下 6,192,224-byte `narration.wav`，四段 MP4 均不存在；
  因而不能把这次退出写成素材上传成功。

## 明日计划

- 以 formal 结果冻结 A1 产品路径和演示口径：Step-Audio 2 mini 负责本地真实出场与
  誊写能力，结构化任务字段进入确定性解析/人工确认链；不再追加额度消耗型样本。
- 从 formal `case-results.jsonl` 提取少量可复核的 clean、noise10、codec32 代表成功与
  失败案例，服务演示与误差说明，不新建训练数据工厂。
- 继续禁止大文件 SSH/rsync 传输；英雄素材如必须上 Spark，应改用赛方允许的国内可
  下载来源或其他合规数据入口，SSH 只保留控制面和小体积证据。
