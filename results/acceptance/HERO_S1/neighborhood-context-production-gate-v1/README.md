# HERO_S1 静态邻域拓扑生产门

日期：2026-07-19

## 结论

纯二维静态邻域链路判定 `NO_GO_PURE_2D_NEIGHBORHOOD`，不替换当前
`reid-hero-s1-closure-multiview-v2` 主工作点。

无 GT 代理在 9 个稳定单实例类别上选择最多 7 个邻居、至少 3 个共同锚点，
proxy winner 为 blend `0.30`。提交前审阅补上两层 fail-closed 语义：无共同锚点
的候选逐字保持 baseline；baseline 高置信 Hungarian 边在全局聚类中优先重建。
修正后 926 条原 automatic 边在三个 blend 中均为零删除。

| blend | 自动/闭环链接 | 澄清 | 新增 automatic | 旧 automatic 删除 | 生产门 |
|---:|---:|---:|---:|---:|---|
| baseline | 994 | 591 | 0 | 0 | 参照 |
| 0.10 | 995 | 592 | 4 | 0 | FAIL：澄清 +1 |
| 0.20 | 993 | 597 | 4 | 0 | FAIL：链接 -1、澄清 +6 |
| 0.30 | 993 | 599 | 9 | 0 | FAIL：链接 -1、澄清 +8 |

0.30 已做两次完整回放，hash 均为
`b6e97cd1ac2996960e935b56f81d365855cda73260d0c23dc0ad50434f396a14`；
确定性通过不等于效果通过。三次运行当时的配置字节已分别封存在
`config-b010.yaml`、`config-b020.yaml`、`config-b030.yaml`，SHA-256 与各自
manifest 完全一致。

## 评估边界

三个 blend 都未通过预先声明的无 GT 生产门，因此正确性修复后没有再次读取
人工 anchor GT。更早的终检发生在两项审计修复之前，已失效，不用于本次选择或
效果声明。下一步若继续，应让视频/高分辨率视觉模型直接判断目标及周围实体，
并用新的未见测试集验收，而不是在当前三段视频上继续调 blend。
