# DAY-06 · 2026-07-18

> 赛程日:D6
>
> 当日主责:AutoTune-v1 一天时间盒——整合 StepFun 3.7 多模态能力作为模型训练一环
>
> 状态:进行中

## 今日目标

- 按验收门修订记录(2026-07-17)执行 AutoTune-v1:StepFun step-3.7-flash 作云端
  调优代理(错误归因/难样本/配对标注)→ 伪标签 → Spark SF1 投影头域内自适应 +
  工作点搜索 → 纯人工 GT 判卷(预算 ≤2 次:中检+终检)。
- 目标门:G2c 完整合并 ≥16/20、G2d R@1 ≥0.90、G2e 澄清 ≤40(终局基线
  14/20、0.8031、1379)。
- 附:落地 R2 回程传输通道(spark→Mac),终结大文件走 SSH。

## 增量 D6-1:AutoTune 工装与云端归因通道(凌晨)

- 工装四件套入库(`395f0db`、`600abf9`):StepFun 客户端补图片通道(单 crop
  368 prompt tokens);`autotune_tutor.py`(归因+配对,内容寻址缓存+调用台账);
  `autotune_pseudo_labels.py`(缝合边+高置信链接+tutor 约束→并查集→SF1 标签
  格式,GT 只出重叠率审计数);`autotune_proxy_eval.py`(内循环代理指标,不读
  GT);阈值扫描接 S5 属性与 min_observations 轴。
- GT 动用台账开档 `results/autotune/GT_USAGE.md`(判卷预算 0/2)。
- **归因通道 8 病理锚点全跑通**(7 解析 +1 正则修复——tutor 在 JSON 内嵌未转义
  引号)。病理分型三类,比终局判卷时的单一"嵌入漂移"假设更立体:
  1. **纯嵌入漂移型**(SF1 主靶):玩具娃娃、跳跳糖——6 碎片被 tutor 判全连通
     同一实例,不变特征清晰(心形眼镜/胸前刺绣;豹子警官/黄色艺术字);
  2. **局部特写碎裂型**:剪刀 A=C=D=E、饼干 A=D=F——特写组只含手柄/刀刃局部;
  3. **烂 crop 型**(`min_quality` 杠杆靶):护手霜 B/C/E/F、梳子 C/E 模糊过曝
     近不可匹配,解释这几锚 R@1 崩塌。
- 对抗性发现记档:tutor 判剪刀组 B 为灰色剪刀、组 F 为另一把浅色剪刀(与 GT
  单实例裁决相左)。GT 冻结不动,仅记为"tutor-GT 分歧"供终局复盘;伪标签只取
  tutor 同物对,不受影响。
- 归因产物:`results/autotune/attribution/`(8 记录 + merge_pairs.jsonl 46 对
  + 英文检测短语建议存档)。配对通道 400 不确定带对并发标注中。
