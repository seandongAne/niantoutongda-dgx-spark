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

## 增量 D5-3：素材疑点用户裁决（07-17）

- **"杯子"=白色马克杯**：确认为锚点。旁白缺独立介绍行，按 GROUP 证据
  优先级走用户轻确认补组（"杯子"组 = 马克杯+水壶+咖啡罐）；s1 首跑产出
  澄清与实体 id 后按需回填 `fixtures/hero_s1/confirmations.json`，
  不伪造旁白行。
- **书=三本统称**：同去向、一起打包；GT 对账按一组处理，不逐本拆锚点。
- **洗手间不补拍**：旁白去向为洗手间的四件（防晒霜/发卡/梳子/护手霜）
  无匹配登记区域，判卷按降级口径如实记录（G5a 该组预期例外），
  不作为区域登记缺失退回素材。
- 裁决已同步写入 items.json/regions.json/vocab.json 注释；零食袋外观
  对应关系随后由队友实物照片确认(山楂棒=白色扁平小袋,05e0353)。

## 增量 D5-4：G0 三段快检——素材过关，无需补拍（傍晚）

- 旧屋三段以 hero-s1-vocab1 的 21 条主短语过真实管线
  （`results/hero_s1/g0/{old_1,old_2,old_3}/`，日志 `logs/g0_hero_s1.log`）：
  关键帧 322/339/439，检出密度 12.8-13.5/帧，长轨(≥4帧) 246-367 条/段。
- **G1a 前景乐观**：全部 20 类锚点词在三段中均有多条长轨命中，无死锚点；
  最弱类 hair_clip 也有 5-9 条长轨（与物品照扫描结论一致，重点盯）。
- 碎轨比 68-78%（1 帧碎片/全轨），主要集中在多实例小件（铅笔×4、
  跳跳糖袋）与已知过召回词；同视频 stitch + 低证据吊销提问权机制在射程内。
- 已知污染按预期重现：跨 prompt 拼接畸形标签（纯召回内部键，不上屏）；
  泛化词过召回（paperback story book 30-40 条长轨 >> 3 本锚点书，旧屋
  或有非锚点书籍；cloud print tin container 15-21 条偏多）——交由 GT
  对账与澄清封顶消化，不动词表冻结原则。
- 判定：**素材质量过关，拍摄合同兑现（慢平移/停留/顺光可检），不触发
  §五自查退回**；s1 发射只剩人工门（听校/vocab 裁决/regions 确认/vLLM）。

## 增量 D5-5：s1 主链发射两连挂——断连免疫补上最后一块（晚）

- 人工门全清后 s1 主链首发（W1 工作点，`configs/hero_pipeline_s1.yaml`）：
  healthcheck 秒过，但管线在 ingest 发射调用里**空转 33 分钟**——远端
  ingest 实际在正常跑，发射 ssh 的回包丢失，而 `ssh()` 的 `subprocess.run`
  无超时，管线永远进不了轮询循环。修复一（`b825191`）：120s 超时斩断，
  超时合成 254 且**不自动重试**（区别于 255：假死时远端可能已执行，盲目
  重发=双重发射）。
- 二次发射按新逻辑 120s 快败，暴露真凶：解剖远端进程链，外层 shell 停在
  `do_wait`（PPID=1 仍不退出）——旧 `nohup ... & echo launched` 发射形状下
  **远端 shell 会陪跑整个长任务**，发射 ssh 要等 ingest 跑完才返回。简化
  命令（`sleep`）不复现；dev-fixture 联调未暴露纯因各阶段秒级完成，"挂
  几秒"不可见。长任务才现形=最危险的一类潜伏 bug。
- 修复二（`afb13db`）：发射改 **`setsid -f` 前台形状**——长任务毫秒级
  过继 init、自建会话（实测 PPID=1/SESS=自身），发射命令前台秒回；
  发射连接层失败(254/255)时以新连接核实远端日志（发射前已 rm）已建
  则转轮询，自愈且杜绝双重发射。第三次发射成功，管线首次真正进入
  轮询循环。
- 代价：两只孤儿 ingest 各跑 33/10 分钟后弃杀（半截产物删除、日志留档
  `hero_ingest.log.hang1/.hang2`），净损 ~50 分钟 GPU；换来发射路径
  真正断连免疫。监控侧同修：主链发射改 `python -u`（stdout 块缓冲曾把
  监视文件憋成空文件）。
- 教训：①远程编排的每一次 `subprocess.run(ssh)` 都必须有超时——"轮询
  免疫断连"不够，**发射调用本身就是单点**；②非幂等远程命令的连接层
  失败禁止盲目重试，正确动作是新连接核实远端状态；③秒级 fixture 联调
  验证不了"长任务才现形"的会话生命周期 bug，联调至少要含一个分钟级
  阶段。

## 增量 D5-6：s1 全链耗时账——六小时里的三个结构性漏洞（深夜）

发射到 S5 收口共约 6 小时（08:07→17:10 UTC），其中**有效 GPU 工时
约 2 小时 53 分**，其余全是三个此前不可见的结构性漏洞的代价：

| 时段(UTC) | 事件 | 耗时 |
|---|---|---|
| 08:07–08:40 | 发射#1：ssh 假死无超时,管线挂死(漏洞一) | 33min 空转 |
| 08:41–08:52 | 发射#2:120s 快败→解剖:nohup& 远端 shell 陪跑长任务(漏洞二) | 11min |
| 08:52–10:59 | 发射#3(setsid):ingest 实扫 old_1/2/3 = 2190/2292/2856s | **122min 有效** |
| 10:52 | 本地 7200s 阶段超时自杀(真实 ingest ≈124min>2h),远端健康续跑 | — |
| 11:02–11:04 | 发射#4:`--adopt-stage ingest --timeout 14400` 无损接上 | 2min |
| 11:04–11:50 | S5 属性:4629 轨 4907 调用,4628 OK/1 failed/97 升级/API 0 错 | **51min 有效** |
| 11:50–16:55 | S5 exit 1(1 轨失败=阶段致命,旧语义),done 不落,管线空轮询(漏洞三) | ~5h 空等 |
| 16:55–17:10 | 诊断+S5 容忍门修复(`112fade`)+缓存重跑(4628 命中) | 15min |

- 漏洞三与今晨 A1 formal 的 scorer 崩溃同构：**个别坏样本不得让整个
  批次失去结论**。S5 新契约=失败轨如实落盘(EXTRACTION_FAILED,下游
  按 missing 语义剔除)+`--max-failed-rate`(默认 1%)容忍门,超限才
  阶段致命。失败样本:`old_1_t1287` 六次调用均产不出合法 schema。
- ingest 单段 ~37-47min(40 短语 14 批次+自适应切片,tiled_kf=12),与
  G0 快检(21 短语)不可比;**长阶段超时预算按 4h 起**。
- 修复链完整落库:`b825191`(ssh 超时)→`afb13db`(setsid 发射+核实
  自愈)→`427a812`(--adopt-stage 收养)→`112fade`(S5 容忍门);至此
  发射/轮询/编排器重启/部分失败四个环节全部免疫。

## 增量 D5-7：GT 到手当夜终局判卷——白名单无弹药，AutoTune-v1 立项（深夜）

- 用户目检确认件到手(20 锚点/364 轨,清洗 dev_a localStorage 残留 61 键入库)。
  首轮判卷(hero 门 16/20, 0.90):W1 12/0.771,W2 14/0.803,误合并均 0 → W2 胜。
- §4 校准迭代:stitch 双探针(0.83→13/0.8115、0.87→14/0.7938)证**白名单无弹药**,
  维持 0.85;封顶已 2。终局落档 `terminal-verdict-2026-07-17.md`:G2a/b/f 过,
  G2c/G2d/G2e 未过如实入档,病理=小件图案面嵌入漂移(玩具娃娃 16/26、咖啡罐
  25/40 最弱),距 G2c 差 2 锚点、G2d 差 0.10。
- 战略裁决(用户,经 codex 对比综合):①验收门直接修订不另立 v2("先射箭再画靶");
  ②数据边界改口——演示运行时零云依赖,开发期允许脱敏 crop 送 StepFun(PII 拍摄
  期已规避);③**AutoTune-v1 批准 D6 时间盒**:云端调优代理(step-3.7-flash 错误
  归因/短语/难样本/搜索建议)→Spark 工作点搜索+SF1 投影头域内自适应→纯人工 GT
  终局重判,口径"调优执行全自动,验证真值独立";④GT 使用契约=尺子和病理报告
  不是教材(禁训练/禁送云/禁内循环,判卷预算 ≤2 次)。
- 当夜另修:旁白解析器适配真实 A1 句式(全角逗号/这是前缀/同段目标+搭子,096dcb7,
  25/25 五要素齐)→GROUP 瓶颈移至实体歧义(3437 实体淹没名字空间,结构解=锚点
  认定);anchor_gt_eval 门限参数化;确认页 crops/ 硬链接修复(4332 张)。
