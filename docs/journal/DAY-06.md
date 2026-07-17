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

## 增量 D6-2:R2 回程通道——大文件传输彻底告别 SSH(清晨)

- 动因:D5 拉回 1.1GB crops 走 SSH 耗近一小时;赛方明令大文件不走 SSH。素材日
  已验证下行(Mac→R2→spark r2.dev 并发拉),本日补上行(spark→Mac),自此
  SSH 只留控制面与小体积证据。
- 机制(`scripts/pull_results_r2.sh` + `infra/r2-relay/`):
  1. Mac 用 wrangler OAuth 部署绑桶 Worker,注入**本次传输专用一次性随机 token**;
  2. spark `tar+zstd -T0` 打包 → 64MiB 分块 + sha256 清单 → `curl -T` 4 路并发
     PUT 到 Worker(**节点全程零凭据**,只见限时 token;曾被入侵机器的凭据纪律
     不破);
  3. Mac wrangler API 6 路并发拉回 → 逐块 SHA256 校验 → cat 重组 → 解包;
  4. trap 兜所有退出路径:桶内对象即删、token 即作废(disabled 重部署)、远端
     /tmp 会话目录清理。
- Worker 侧防御:token 比对 + 只许写 `xfer-` 会话前缀(防覆盖桶内素材对象)+
  路径穿越拒绝。
- 旧 `pull_results.sh` 加体积预检门:rsync dry-run 测增量,>50MB 拒绝并指路
  R2 通道——1.1GB 级事故从机制上灭绝。
- 三镜头对抗审查(壳层/凭据/断连故障,21 agent)实锤 **2 critical + 12 major**:
  远端 tar|zstd 缺 pipefail(截断归档可静默通过全套 SHA 校验)、600s 击杀路径下
  一次性 token 变长命 token、cleanup 网络操作吞错、set -e 在赋值管道里把回退
  分支变死代码(两处)、上传成功判定与 ssh 退出码耦合、tmpfs 暂存钉死内存等。
  全部修入;核实员并实测桶 r2.dev 公读已关(暴露面比预设小)。
- **E2E 首跑即证伪 Worker 架构**:spark 侧 workers.dev 被 DNS 污染(解析到
  face:b00c Facebook 段)+ SNI 阻断,钉真实 CF IP 亦超时;而 S3 端点
  r2.cloudflarestorage.com 实测可达(HTTP 400 匿名拒绝)。**v2 转预签名架构**
  (`97da905b`):Mac 持 R2 S3 密钥生成限时单对象预签名 URL,spark 只见 URL——
  无凭据落节点、无常驻端点、URL 自动过期,token 生命周期问题连根消失;
  Worker 云端与仓库双退役;桶生命周期规则 xfer-ttl(1 天过期)兜底孤儿对象。
- **E2E 验证通过,通道转正**(数据所有者创建桶限定 R2 令牌,只入本地 .env;
  凭据粘贴带包裹字符,净化脚本按 32/64 hex 长度校验修复):
  - SigV4 自测:PUT 200 / GET 回读一致 / DELETE 204;
  - reid-w2(7.8M→zstd 后 <1MiB,单块):逐字节一致;
  - old_1/keyframes(94MB JPEG 不可压,12 块):**46.5 秒全程逐字节一致**,
    有效吞吐 ~2MB/s ≈ SSH 实测(17KB/s)的 120 倍;1.1GB 级拉取 ~10 分钟。
  - trap 清理实测:失败路径(workers.dev 探测失败)与成功路径均正确善后。
