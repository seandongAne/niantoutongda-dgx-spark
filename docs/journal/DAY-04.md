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
