# DAY-08 · 2026-07-20

> 赛程日:D8
>
> 当日主责:公开仓库入口文档与远端发布
>
> 状态:已收工

## 今日目标

- 为公开仓库建立可独立阅读的 `README.md`,用统一入口说明产品问题、四 Agent
  架构、已验证结果、快速开始、Spark 执行纪律、仓库结构与设计边界。
- 将首页数字逐项绑定到已入库的机器可读证据,避免把受控英雄任务写成通用产品能力,
  也不把可选的搬后照片复核表述为已完成物理验收。
- 验证 README 中的本地测试、主链 dry-run 与相对链接,仅提交本次文档和当日日志,
  不混入工作区已有的未跟踪实验产物；完成 `main` 远端发布。

## 关键证据或截图

- 新增 144 行项目首页,覆盖“项目定位 → 系统流程 → 英雄任务证据 → 技术栈 →
  快速开始 → Spark 安全纪律 → 设计边界 → 延伸文档”。技术提交:
  `95d72c64`(`docs: 添加项目 README`)。
- 首页结果表逐项引用正式冻结产物:
  `results/hero/s1-auto-final-v1/inventory/metrics.json`、
  `showcase_metrics.json`、`group/metrics.json`、`spatial/metrics.json`、
  `spatial_score/metrics.json`、`layout/plan.json` 与
  `audit/replay-report.json`;写入口径为 `3306→20`、完整合并 `17/20`、联合
  R@1 `85.79%`、组合覆盖 `20/20`、空间 `2278→168→5`、评分 `5/5`、
  5 个布局分配与严格回放 `PASS`。
- 本地全量后端验证 `370 passed in 9.28s`;README 全部相对链接存在,
  `git diff --check` 通过。主链命令
  `.venv/bin/python scripts/hero_pipeline.py --config configs/hero_pipeline_s1_final.yaml --dry-run`
  成功列出 healthcheck、两段 Spark 任务、拉回、库存、空间评分、组合、布局、
  任务卡、trace、报告与 bundle 共 14 个阶段。
- `git push --dry-run origin main` 先验证 Git 写路径,随后实际发布成功:
  `817e8e96..95d72c64  main -> main`;远端为
  `https://github.com/seandongAne/niantoutongda-dgx-spark.git`。
- 未验证或降级边界:本次没有连接 Spark、没有加载模型、没有改写实验结果；公开仓库
  不能重建未入库的授权原视频。README 明示受控单场景、区域级布局和物理验收默认关闭。

## 失败与教训

- `gh auth status` 显示账号 `seandongAne` 的 CLI token 已失效,但 Git HTTPS
  凭据仍可完成 dry-run 与实际 push。Git 数据通道与 GitHub API/PR 通道必须分别
  验证,不能把一个通道的认证失败直接推断为全部远端发布失败。
- 工作区存在大量未跟踪实验产物；若使用 `git add -A` 会污染 README 提交。
  本次只按明确路径暂存 `README.md` 与当日日志,并在提交前复核 cached diff。
- `make demo` 仍是未实现占位入口,因此 README 没有把它包装成可用命令；正式展示
  采用已冻结离线成果页,完整技术链使用 `scripts/hero_pipeline.py`。

## 明日计划

- D9 按项目纪律汇编并复核 `docs/十日谈_念头通达.md`,将演示视频、最终提交包和
  新增团队证据按真实完成状态补入,不把待办提前写成完成。
- 若新增成片链接或物理验收证据,同步更新 README 的项目状态与边界,再次执行链接检查、
  `make test` 和主链 dry-run；无新证据则保持功能冻结。
- 只有在需要 GitHub PR/API 操作时才执行 `gh auth login -h github.com`;普通 Git
  push 继续单独验证,避免无关认证维护阻塞交付。
