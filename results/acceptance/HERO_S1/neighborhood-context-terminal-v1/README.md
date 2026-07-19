# 已废弃：HERO_S1 静态邻域拓扑早期终检

日期：2026-07-19

此目录中的 `gt-eval.json` 发生在两项提交前审计修复之前：

1. 无足够共同锚点的候选仍被中性值带入分位数重排；
2. baseline 高置信锁只覆盖 Hungarian，没有贯穿后续全局聚类。

因此该 GT 结果已经失效，不得用于选参、效果声明或主链提升。两项修复后，三个
预设 blend 均未通过无 GT 生产门，故没有再次读取人工 anchor GT。最终机器可读
判决见 `../neighborhood-context-production-gate-v1/report.json`。
