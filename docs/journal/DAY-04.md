# DAY-04 · 2026-07-16

> 赛程日：D4
>
> 当日主责：S3 跨视频匹配与 S2.5 检测推理修复
>
> 状态：已完成（S2.5/S3 可复跑基线闭环；人工真值门待办）

## 今日目标

- 启动 S3 跨视频匹配关键路径，用 `dev-a-vocab5` 四视频现有产物先形成可复跑基线，并逐组检查 A–D 四组同类不同实例硬负样本。
- 落地 S2.5 推理侧核心修复，冻结参数后只重跑一次 `dev-a-vocab6`，与 v5 对照。
- 将 `vlm_attributes` 主路切换为 Nemotron VL NVFP4-QAD/vLLM，保留 BF16 fallback；裁撤独立 9B `task_copy` 槽位。

## 关键证据或截图

- 完成结果：
  - `configs/models.yaml` 已将 Nemotron VL NVFP4-QAD/vLLM 标为 `probed_spark` 主路，BF16 保留为显式 fallback；独立 9B 文案模型撤出，任务卡正文继续走确定性结构化渲染，12B 同服务只作可选润色。
  - S2.5-1/2：`fixtures/dev_a/vocab.json`、canonical/category 分层词表编译、同 canonical 别名 NMS 与跨概念重叠保留均已落地并有单测。
  - S2.5-3/5 评测底座：固定 IoU=0.5 一对一匹配与三指标口径；真实 30–50 帧人工框 GT 尚不存在，因此当前不得产出伪造的 v5/v6 三指标或 prompt 优胜结论。
  - S2.5-4/7/8：停留视角 2×2 + 选择性 3×3 tile、GDINO 跨帧 batch、面积×清晰度×完整度 hero 评分已落地。GB10 探针定参为 batch=2；batch=4/8 不再扩大。
  - S3 v5 确定性 baseline 三次复跑 hash 全同：`3b9d88eab7125e7493309d158957446872c3838380cad57678caf0e65d553f60`。621 tracklets → 564 entities，62 自动链接、544 澄清请求；当前是诊断基线，不是 G2 通过。
  - `dev-a-vocab6` 四视频一次性重跑完成并拉回：614 keyframes、6563 observations、945 tracklets、1659s；相对 v5 的 3919 / 621 / 1249s，墙钟为 1.328×，守住 ≤2× 成本门。全部 945 条 tracklet 均有 hero 引用。
  - v6 的 canonical tracklet 出现量显示弱词覆盖面扩大：`security_camera` 从 4/5/5/8 增至 9/11/10/11，`table_lamp` 从 0/1/1/0 增至 7/7/8/9，`luggage` 从 3/2/0/2 增至 11/11/6/10。该结果只证明候选出现量变化，不等于召回提升。
  - S3 v6 三次复跑 hash 全同：`700c121094068dd96ab8de4d2256013ba8ee6887fcef4b4d9504987af71cc613`。945 tracklets → 871 entities，83 自动链接、921 澄清请求；相对 v5，自动链接 +21，但澄清 +377，说明当前主要矛盾已从“弱类看不见”转为“候选增多后碎片和歧义放大”。
  - v6 四类联系表视觉抽查未见明显硬负跨物体自动互并，但仅属无真值抽查：`cabinet` 178 轨中类别污染最重；`water_bottle` 为 72 轨 → 72 entities、零自动链接，主要失败仍是同物漏合并。
  - S3 v6 的 7811 条 candidate 中 `attribute` 分量恒为 1.0、`context` 恒为 0.5，且冻结配置中两者权重均为 0；因此 hero 100% 目前只证明字段覆盖，尚未给跨视频匹配提供可区分属性信号。
  - v5 四组硬负目视草稿覆盖内未发现 opposite-anchor 直接或传递误合并；草稿严格标记 `visual_draft_pending_data_owner`，四组均只能写 `VISUAL_DRAFT_NO_CROSSING_OBSERVED_PARTIAL_MAPPING`。主要失败是同物漏合并：蓝色水壶 6 轨仍为 6 entities，玫红水壶 9 轨仍为 8 entities。
- Commit：`3307d30`、`a7c961a`、`3036331`、`8077b52`、`4930a99`、`df1f2f2`、`dbde058`、`8ab12ac`、`b0ae75f`、`fbc5fd4`、`9a7ceb4`、`7a525f3`、`bcfb6c0`、`542382e`。
- 验收证据：
  - `results/acceptance/S3/s3-v5-8077b52/metrics.json`
  - `results/acceptance/S3/s3-v5-8077b52/hard-negative-audit.visual-draft.json`
  - `results/acceptance/S3/s3-v5-8077b52/contact-sheets/`
  - `results/acceptance/S2.5/dev-a-v5-v6-7a525f3/ingest-diagnostic.json`
  - `results/acceptance/S2.5/tile-ab-v1-7a525f3/ingest-diagnostic.json`
  - `results/acceptance/S3/s3-v6-7a525f3/metrics.json`
  - `results/acceptance/S3/s3-v6-7a525f3/contact-sheets/`
- 日志 / metrics / 截图：
  - v5 S3 指标：`automatic_link_count=62`、`clarification_count=544`、`deterministic=true`、`g2_evaluated=false`。
  - GDINO 真机定参：batch=2 为 1.158×、batch=4 为 1.112×、batch=8 为 0.969×；batch 与逐帧输出结构一致，最大分数差 `0.000671`、最大框差 `0.0041px`，在冻结的 `1e-3 / 0.5px` 决策容差内。
  - tile 首探针 147 框，经“边缘截断剔除 + 全图面积≤12% + 每 canonical 最多 3 个补充”降到 38 框；完整整帧检测 18 框未丢。
  - 首个 v6 部分尝试在 v1 暴露 `tiled_kf=0` 后主动停止并保留为
    `ingest_a_v6_aborted_stationary0`；修复后完整 v6 按
    `configs/ingest_dev_a_v6.yaml` 以 nohup 完成四段，逐段墙钟为
    469/400/328/462s，tile 帧为 9/11/6/12。
  - v1 的同代码 tile A/B：相对未触发 tile 的 aborted 产物增加 54 observations、
    1 tracklet，墙钟 +97s；新增 tracklet 仅为 1 条 `stuffed_animal`。这是无真值诊断，
    只能证明 tile 成本有界，不能据此宣称召回收益。
  - v5/v6 诊断文件显式写入 `hardval_metrics_evaluated=false`；轨迹数、候选数和
    canonical 出现量均不得替代锚点召回、碎轨率或 FP/frame。
  - 本地完整测试：`64 passed in 1.01s`；YAML 与 Python 编译检查通过。
- 未验证或降级边界：
  - 17-anchor 完整机器真值仍缺失；S3 v5/v6 均不能计算 Recall@1、15/17 完整合并或官方四组 hard-negative PASS。
  - S2.5 hardval 人工框尚未完成，故三核心指标对照仍被数据门阻塞。
  - 945/945 hero 仅完成引用覆盖；17-anchor hero 清晰度/完整度人工质量抽检尚未做。
  - S3 v6 联系表已生成，但无 anchor→tracklet 真值，故四组硬负仍不是正式 PASS；
    `g2_evaluated=false` 保持不变。其 manifest 的 `code_commit` 仍为 `unknown`，目录后缀与
    本地证据提交不能替代生成时 provenance，后续需修生成器。
  - S2.5-6 云视觉教师/增强数据工厂不得在未获本批影像出境批准时上传家庭 crop；不阻塞 v6 主链。

## 失败与教训

- Codex 桌面端在约十五分钟内多次表现为界面挂起；主机内存、swap、磁盘和后端进程都无资源瓶颈，macOS 日志反而记录 ChatGPT 主线程可能导致 UI unresponsiveness 及通知/scene XPC 异常。重启后最后一次多子代理并行完整收敛，故本轮没有因“可能是网络”掩盖客户端问题。
- GDINO “128GB 放得下大 batch”不等于大 batch 更快。batch=8 真机反而慢于逐帧；必须先扫 2/4/8，再把实测最优 batch=2 写进冻结配置。
- 原始停留帧 tile 在低阈值下把单帧候选从 18 放大到 147；多尺度召回若没有面积、截断和每类上限，会把 S2.5-4 变成误报放大器。修复后才允许进入 v6。
- 第一个 v6 尝试的真实日志为 `tiled_kf=0`。继续跑完只会得到一个不含 S2.5-4 的伪 v6，故在 v1 后停止、产物非破坏性改名留档。四视频 2fps 光流复核证明严格 2s 低运动段只有 0/0/1/0 个；最终采用显式的低运动 10% fallback（2s 时间 NMS、每段最多 12），无模型复验选中 9/11/6/12 帧。
- 无 anchor→tracklet 真值时，算法链跑通和三次 hash 一致只证明“可重复”，不能证明“正确”；目视草稿也不能冒充 data-owner ground truth。
- v6 将 observation 增加 67%、tracklet 增加 52%，但 S3 澄清请求也增加 69%。
  这说明继续降低检测阈值或扩大 tile 只会把噪声推给 S3；下一步应优先做候选门控、
  同视频轨迹缝合和更强属性证据，而不是继续追求候选总量。

## 明日计划

- P0：完成 30–50 帧 task-A hardval 人工框与 17-anchor→tracklet 映射，补齐 v5/v6 三指标和正式四组硬负判卷；在此之前不调整验收结论。
- P1：在 S3 前增加低质量/重复候选门控，并先做同视频短轨缝合；目标是降低 921 条澄清，而不是简单放宽自动合并阈值。
- P1：把已产出的 hero crop 接入 NVFP4 属性抽取，形成颜色/材质/文字标记等多证据原型，再复验蓝色与玫红水壶等同类不同实例。
- P1：修正 S3 缺失属性的默认相似度语义，使 missing/unknown 不再产生恒定 1.0；属性产生方差且有标注验证后，才从 0 调高 `attribute/context` 权重。
- P2：hardval 证明 tile 有真实收益后，才保留按类/按场景触发；否则缩减或关闭低运动 fallback，避免为候选膨胀支付 32.8% 墙钟成本。

## D4 午后增量(stitch + GT 物料,commit 4482129)

- **属性链路根因闭合**:S3 候选 `attribute≡1.0` 的机制 = v6 tracklet.attributes 只有
  label/hero_ref/hero_score/hero_scoring_version 四个流水线元数据键,matcher 忽略集漏了
  常量版本串 `hero_scoring_version`(145f3fa 已修,回归测试);`context≡0.5` 为硬编码占位;
  两权重在 v1 配置本就是 0。更根本:backend 无任何代码把 Nemotron/S5 属性写入
  tracklet.attributes——属性方差是接线工程,不只是相似度语义修正。
- **S3 v2 预处理落地**(e75bded + 4482129):同视频 stitch(共现硬否决 + 簇级 veto,
  代表 id 取成员最小者,对外全部展开回原始 id 空间)、低证据标记、澄清互选封顶。
  v1 配置不带新节时行为与冻结基线逐字段一致(真机 diff 验证:7811 candidates 仅
  components.attribute 显示值 1.0→0.5,entities/clarifications/accepted-links 逐字节相同;
  新合法基线 hash `e0fdc61b…`)。
- **stitch 参数探针**(`results/acceptance/S3/stitch-probe/probe-v6.json`):v6 同视频同标签
  12743 对,余弦 p50=0.31/p99=0.816;0.80 阈值 108 次共现否决 vs 42 次合并 → 高余弦对多数
  是共现的不同物体,共现否决是承重墙;冻结 v2 = stitch 0.85(23 merges)。观测数直方图
  {2:238,3:153,4:104,5+:450},无单观测轨。
- **关键教训(一次 A/B 换来的)**:`min_observations=3` 若做成硬剔除,丢 13 条 0.865–0.954
  的自动链接,全部是弱类(security_camera×3/storage_box+玩具收纳×5/luggage×2/
  stuffed_animal×2)——两观测轨恰是 v6 找回的弱类召回本身。改为"保留匹配与自动链接
  资格、仅吊销提问权"后 13/13 恢复。
- **S3 v6 A/B 终版**(均三次 hash 全同):

  | 指标 | v1 控制 | v2 处理 |
  |---|---|---|
  | 澄清请求 | 921 | **564 (−38.8%)** |
  | 自动链接 | 83 | **85** |
  | matched entities | 59 | 61 |
  | 低证据摘除澄清 | — | 242 |
  | 互选封顶摘除 | — | 65 |
  | stitch 合并 | — | 23 组(945→922) |

  match/margin/new 阈值与 v1 逐字段一致,未用降阈值换数字。
- **GT 物料齐备待人工**:hardval 40 帧(焦点分+时间步进混采,`fixtures/dev_a/hardval/`,
  含 gt.skeleton + v6 predictions 746 条 + 离线标注器 annotate.html,评测回环已对
  hardval_eval.py 验证);17 锚点候选审阅表 + 预填 review JSON
  (`results/acceptance/S3/anchor-review-v6/`,候选=同 category 全集,无相似度筛选);
  四组硬负真值在锚点确认后自动导出。22 个 stitch 合并组(23 次合并)目检表随 v2 产物留档,目检未见硬负互并(无 GT,不算 PASS)。
- **P2 顺手修复**:acceptance manifest `code_commit=unknown` — deploy.sh 现写 COMMIT 戳,
  reid_task 回退读取;本批 manifest 已带真实 commit。
- 本地测试 75 passed;sp0 三次核心探针产物入库(c5b3e1d)。

## D4 夜间增量:真值落地,首批正式判卷(commit ba90e1f→ec4de2d)

- **数据所有者真值双双落地**:17 锚点确认(161 轨,零重复归属、零硬负冲突,
  17/17 跨视频)+ hardval 40 帧 71 框。数据门解除。
- **hardval v5/v6 首次真实对照**:召回 0.464→0.710(+24.6pp;security_camera
  0/9→5/9、water_bottle 2/8→8/8、玩具收纳 3/6→6/6),但碎轨率 0.037→0.206、
  FP/帧 8.5→17.4。S2.5 双门诚实结论:目标失败指标(弱类召回)大幅改善,
  碎轨/FP 回退,账单已转 S3 前处理与属性链。luggage 0/2、tumbler 0/1、
  table_lamp 1/4 仍死——进阶跃词表工厂。
- **G2 首次正式判卷**(v1 与 v2 运行):四组硬负 **4/4 正式 PASS**、高置信误合并
  **0**(红线守住);Recall@1 0.81(门 0.85);完整合并 **2/17**(门 15)。
  机制定位:65 条同锚点对分数 ≥0.86 但只有 16 条成链——margin 掐死四分之三
  合格链接;同锚点分数 p50=0.673 vs 跨锚点 0.534,嵌入可分性是结构性天花板。
- **GT 授权的阈值扫描**(12 组合 match×margin):全网格误合并=0、危险澄清恒=4;
  完整合并上限 4/17(margin=0)——**校准无法补足 15/17 的缺口,属性接线才是
  结构解**。帕累托点 m0.84-g0.00:链接 85→156(+71 无丢失)、澄清 564→525。
  **v3 冻结待 Sean 目检 71 对新增链接**(`new-links-m084g000.jpg`;锚点 GT 只
  覆盖 161/945 轨,新增链接的非锚点部分未被测量)。
- **阶跃 Step Plan 启用**(用途①②,均 dev-time、零家庭素材出境):
  - ①A1 预热:客户端补 tts 动词;合成旁白→stepaudio-2.5-chat 抽取回环一次通,
    3/3 物品 owner/来源/去向/打包组全对;协议 v1 冻结
    (`results/stepfun/a1_warmup/PROTOCOL.md`),本地 Step-Audio mini 直接继承。
  - ②词表工厂:step-3.7-flash 产 48 个外观化检测短语候选(六个弱词),
    经 S2.5-3 prompt_search 用新 GT 打分后才可入词表。token 用量已记账。
- 判卷工具链固化:`anchor_gt_eval.py`(冻结 gate 判卷器)、
  `reid_threshold_sweep.py`(GT 授权扫描)、`link_pair_sheet.py`(链接目检表)。
- 下一战排序(依据今日证据):P1 属性接线(S5 Nemotron→tracklet.attributes→
  matcher,结构性缺口唯一解)> v3 阈值冻结(等目检)> 词候选 GDINO 打分。

## D4 深夜增量:产品方向拍板 — 家居视觉校准工厂

- **数据难度归因确认**:碎轨 0.206 / FP 17.4 是儿童房杂乱(遮挡/堆叠/同款多件)
  直接买单;完整合并 2/17 的嵌入天花板则与场景无关——两笔账分开记,
  dev_a 定位为压力测试集而非失败素材。
- **Sean 拍板产品方向升级**:不让用户逐房间调 GDINO;开发期以
  Step 3.7 Flash(教师:词表体系/易混组/失败归因)+ Step Image Edit 2
  (困难版本工厂)+ prompt_search + 真实留出集裁决,持续产出版本化能力包
  household-pack-vN;运行时纯本地加载 pack,用户侧 = "自动盘点 + 轻确认"。
  专档 `docs/家居视觉校准工厂_设计.md`(含四条护栏:合成只排序真实才发布/
  保身份校验/pack 只管检测层/红线继承,家庭素材彻底不出境)。
- **赛内最小切片**:48 候选打分即工厂 v0 第一炉;Image Edit 一小炉验保身份
  校验;dev_a 词表+工作点重组为 household-pack-v0。不改变下一战排序
  (属性接线 > v3 冻结 > 词候选打分)。
- **dev_b 演示级素材计划**同日落档(设计文档 §7):慢平移/物品停留/远近两圈/
  留空隙/顺光 + 顺录旁白;一份素材三用(演示对照叙事/工厂第二留出集/
  A1 真实噪声复测),穿插 S5 跑批空档执行。
