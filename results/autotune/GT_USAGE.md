# 364 轨人工确认集动用台账(AutoTune-v1 期)

> 范围说明：本表只记录 AutoTune-v1 的 GT 动用与当时单工作点读数，
> 不是当前成果页终值。当前统一口径为：**完整合并 `17/20`；多策略
> 联合 R@1 `495/577 = 0.8579`**，见
> `results/acceptance/HERO_S1/multiview-closure-attempt/README.md`。

契约(验收门修订记录 2026-07-17):GT = 判卷与错误归因专用;禁作训练标签/伪标签
来源、禁送云、禁作内循环停止门;AutoTune 期判卷预算 ≤2 次(中检+终检),每次记档。

## AutoTune 前(§4 校准迭代,已在 terminal-verdict-2026-07-17.md 记档)

- 2026-07-17 W1/W2 双判卷(`gt-eval-w1w2`)
- 2026-07-17 stitch 双探针判卷(`gt-eval-stitch-probe`)

## AutoTune-v1 期动用记录

| # | 日期 | 用途类别 | 说明 | 判卷预算 |
|---|---|---|---|---|
| 1 | 2026-07-18 | 错误归因(选样) | 按终局病理选 8 锚点(04/05/06/09/12/14/15/17),读其 confirmed 集在 reid-w2 实体中的碎裂分组,生成 attribution contact 组。**送云内容仅为 crop 与模型自身实体分组,GT 归属不送云** | 不占 |
| 2 | 2026-07-18 | **中检批判卷** | 18 组合扫描(sf1auto 嵌入)先经代理指标(tutor 一致率/澄清数,不碰 GT)收敛到 4 头部候选,单批一次判卷:m0.82-g0.00→完整 14/R@1 0.8083/误合并 0;m0.84-g0.00→12;m0.82-g0.02 与 m0.84-g0.02→11;R@1 四者同(阈值不改嵌入)。证据 `results/autotune/gt-midcheck.json`。选定工作点 m0.82-g0.00-o5(o 轴为纯澄清杠杆,不碰 GT 维度) | **1/2** |

| 3 | 2026-07-18 | **终检批判卷** | 主链 final 跑(`results/hero_s1/reid-final`,config reid-hero-s1-final-autotune-v1)单批一次判卷归档:完整合并 14/20、R@1 0.8083(468/579)、高置信误合并 0、硬负 PASS——与中检 m0.82-g0.00 完全一致(reid 确定性验证)。证据 `results/autotune/gt-terminal.json` | **2/2** |

判卷预算使用:2/2(预算用毕;AutoTune-v1 期 GT 动用通道关闭)。
