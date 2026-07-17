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

## 增量 D5-2：素材日——R2 HTTP 中转通道与三 TODO 落地（下午）

- **素材到手**：`~/Downloads/Dev B` 七文件全为 1080p60 H.264——旧房间三段
  （162s/170s/220s）、新家两光照（`new_1` 开灯 104s、`new_1_natural_lighting`
  111s）、`item_collections.MOV`（物品摆拍 14s）、`narration.m4a`（217s）。
  本地无损转封装为 mp4（流拷贝零重编码）、旁白转 16kHz 单声道 wav。
- **SSH 大文件传输尝试中止**：实测吞吐 ~17.5KB/s，
  1.84GB 需约 29 小时；用户裁决保原画质、走第三方 HTTP 中转，压缩降质路线否决。
- **通道实测**（spark 侧）：Google/Drive 全阻断；github.com 超时；
  `release-assets.githubusercontent.com` 可达但 1.3KB/s 不可用；
  `speed.cloudflare.com` 1.95MB/s、ModelScope range 1.78MB/s 可用。
  落地 **Cloudflare R2 私有桶 + r2.dev 直链**：Mac 上行 16MB/s（200MB/12s）；
  r2.dev 单连接限速 ~100-600KB/s、并发可叠加（4 连接聚合 ~1.5MB/s），
  故 200MiB 分块 ×12 + `xargs -P 6` 并发 + `curl -C -` 断点重试拉取，
  节点侧 `cat` 重组后按 `SHA256SUMS` 校验；传毕删桶内对象。凭据全程只在
  Mac（wrangler OAuth），节点只见限时公开直链，符合凭据纪律。
  旁白 wav（6.9MB）循 A1 音频先例走 SSH 补传完成。
- **旁白 TODO**：新增 `scripts/a1_hero_transcribe.py`（报文构造逐字复用
  `a1_stepaudio_local.py`，PROTOCOL.md 本地誊写口径：零-shot 贪心）。
  v1 静音切分失效（top_db=40 仅 3 段、亚秒碎段幻化系统提示词），重写为
  停顿分组切分（间隙 ≥min_gap 断句 + 超长段最长间隙递归二分 + <0.8s 碎段
  丢弃），参数空跑扫描后取 top_db=25/min_gap=0.5 → **30 段全部成句**，
  五要素句式完整。碎行合并后 25 行落 `fixtures/hero_s1/transcript.txt`
  草稿（待人工听校）；逐段时间戳留 `results/hero_s1/a1_transcribe/`。
- **词表 TODO**：`item_collections.MOV` 逐帧目视起草 items.json，与誊写
  对账后 v2 定稿 20 类（零食拆为跳跳糖/山楂棒/饼干三件；name_zh 取旁白
  用词）；StepFun 候选 115 短语（云端只见纯文本清单）；28 帧物品照经 R2
  通道上节点，`word_spark_factory` scan-only 免凭据发射（无 GT，出死词
  摘要待人工裁决）。
- **区域 TODO**：`new_1` 抽帧起草 `fixtures/hero_s1/regions.json` 七区域
  （沙发座面/长凳/书桌/花布桌/墙搁板/斗柜/展示柜），evidence_refs 带
  时间戳，待人工确认。
- **对账疑点（需队友确认）**：①旁白两次"和杯子打包在一起"但"杯子"（马克杯）
  未单独旁白，G1d 覆盖风险；②物品视频三本书、旁白仅一本；③防晒霜/发卡/
  梳子/护手霜去向为"洗手间"，而 new_1 巡拍仅客厅、无洗手间区域可登记,
  G5a 指派一致性有隐患；④三种零食袋与外观对应为推断。
- 教训：`pkill -f <脚本名>` 与同一 ssh 命令行内的重启命令同发会自匹配杀掉
  承载 shell（表现为 ssh 255 貌似断连）——kill 与 relaunch 必须分连接执行。
- Commits：`c88c502`（夹具草稿+誊写工具）、`d10bdae`（停顿切分修复）、
  `0519160`（items v2 对账+候选重产）。

## 明日计划

- 以 formal 结果冻结 A1 产品路径和演示口径：Step-Audio 2 mini 负责本地真实出场与
  誊写能力，结构化任务字段进入确定性解析/人工确认链；不再追加额度消耗型样本。
- 从 formal `case-results.jsonl` 提取少量可复核的 clean、noise10、codec32 代表成功与
  失败案例，服务演示与误差说明，不新建训练数据工厂。
- 继续禁止大文件 SSH/rsync 传输；英雄素材如必须上 Spark，应改用赛方允许的国内可
  下载来源或其他合规数据入口，SSH 只保留控制面和小体积证据。（D5-2 已落地：
  R2 HTTP 分块中转为素材标准入口，SSH 仅控制面+小证据。）
- 词表扫描死词摘要人工裁决 → `fixtures/hero_s1/vocab.json`；transcript 人工
  听校；regions 与洗手间疑点队友确认；S5 vLLM 起服务后按
  `configs/hero_pipeline_s1.yaml` 发射 s1 主链。
