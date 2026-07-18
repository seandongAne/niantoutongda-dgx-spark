# DAY-06 · 2026-07-18

> 赛程日:D6
>
> 当日主责:AutoTune-v1、基础完赛与英雄任务全自动空间技术闭环
>
> 状态:进行中

## 今日目标

- 按验收门修订记录(2026-07-17)执行 AutoTune-v1:StepFun step-3.7-flash 作云端
  调优代理(错误归因/难样本/配对标注)→ 伪标签 → Spark SF1 投影头域内自适应 +
  工作点搜索 → 纯人工 GT 判卷(预算 ≤2 次:中检+终检)。
- 目标门:G2c 完整合并 ≥16/20、G2d R@1 ≥0.90、G2e 澄清 ≤40(终局基线
  14/20、0.8031、1379)。
- 附:落地 R2 回程传输通道(spark→Mac),终结大文件走 SSH。
- 在不新增视频、不重跑 ReID 的前提下,以独立目标视图和全局一对一门完成自动五区,
  替换 D5 的视觉裁定生产路径,并复跑 3306→20、组合、布局、任务卡与审计主链。
- 形成可供队友直接整合的参赛技术介绍,以正式主链与实测指标为事实底座,TensorRT
  保持部署测试态口径。

## 关键证据或截图

### 增量 D6-1:AutoTune 工装与云端归因通道(凌晨)

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

### 增量 D6-2:R2 回程通道——大文件传输彻底告别 SSH(清晨)

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

### 增量 D6-3:工作点扫描、两次判卷与主链定版(上午)

- **伪标签 → SF1 → 18 组合扫描**:tutor 同物对 + 缝合边 + 高置信链接经并查集出
  117 身份/511 样本(GT 重叠率仅 12.52%,审计数),SF1 三种子门 PASS(R@1
  +0.0107、margin +0.0031——比 dev_a 先例温和);扫描轴 match{0.82,0.84,0.86}
  × margin{0,0.02} × min_obs{3,4,5}。
- **代理指标先行收敛(不碰 GT)**:min_obs 是纯澄清杠杆(o3→o5 澄清 -45%,
  实体/链接/tutor 一致率纹丝不动);tutor same_recall 强烈偏好宽松 match
  (0.223@0.82 vs 0.118@0.86),margin 0.02 全线丢召回、护栏几乎不涨。
- **中检批判卷(预算 1/2)**:4 头部候选单批判卷,m0.82-g0.00 胜出——完整
  14/20、R@1 0.8083、误合并 0;margin 0.02 实测完整合并 14→11 而误合并保护
  0→0,代理指标方向被 GT 证实。R@1 四候选同值:阈值不改嵌入,0.8083 全部
  来自 SF1 投影。
- **定版与终检(预算 2/2)**:终选工作点 `m0.82-g0.00-o5` 落冻结配置
  (`configs/reid_hero_s1_final.yaml`),主链 final 跑(ingest/attributes 输入
  未变免跑),终检判卷与中检完全一致——reid 确定性顺带得证。台账关账。
- **AutoTune-v1 终账**(基线→定版):R@1 0.8031→**0.8083**(+0.5pp,SF1 域内
  自适应),完整合并 **14/20 持平**,高置信误合并 **0 持平**,澄清
  1379→**677**(-51%,min_obs 杠杆)。G2c/G2d 预注册门(16/0.90)未达:
  嵌入天花板仍在,12.5% GT 重叠的伪标签撬不动 anchor 级完整合并;按 07-17
  验收门修订裁决如实记档,不粉饰。
- **马克杯轻确认回填**:"杯子"组 = 马克杯+水壶+咖啡罐(07-17 裁决),三实体
  hero crop 逐张人工核图后写 `fixtures/hero_s1/confirmations.json`;顺手揪出
  词表病理——reid 词表"水壶"名下 5 实体全为误检(其一是卷笔刀),真品被 VLM
  命名"水瓶(粉色)"(粉色保温杯青绿翻盖,与 items.json 外观档案吻合)。
  groups 1→2、taskcards 2、trace PASS。
- **附带修真一道门**:macOS openrsync 在 dry-run 下 `--stats` 恒报 0 transferred,
  `pull_results.sh` 的 50MB 门自建成起从未生效;改 `--out-format '%l %n'` 逐文件
  求和,实测对 138MB 增量正确拒绝。sweep 全量产物(139MB)走 R2 通道实战首拉,
  zstd 压至单块、秒级到本地。

### 增量 D6-4:交付三件套备便(上午)

- **验收一键就绪**:`fixtures/hero_s1/acceptance.template.json` 按 s1-final 真实
  卡片/区域/实体 ID 预填(present 预填 false=失败安全,不填表跑出的是 FAILED
  而非假 VERIFIED),已过 AcceptanceManifest 合同校验;拍摄/填表/发射三步指引
  `docs/验收照片指引_hero_s1.md`。照片到手后:填表 → verify/strict 翻 true →
  `--from-stage verify`,G7a/G7b 同跑取证。
- **演示脚本底稿** `docs/演示视频脚本_v1.md`:9 镜头 100–120s,素材指针全部指向
  已归档产物,双生态段位对齐评分标准(NVFP4/vLLM 15× vs StepFun 调优代理),
  诚实边界段预答未达门;录屏采集清单机器侧可先备。
- **十日谈重构并填至 D6**:海事时代骨架换成真实弧线(生死门→换题→链路与真值
  →StepFun 3.7 整合训练一环→演示固化占位→结果与边界),口径=错误归因/难样本
  判定/伪标签域内自适应,数字与判卷档案逐一对应。

### 增量 D6-5:基础完赛——赛方参考代码全链路跑通(下午)

- **背景**:赛方宣布基础完赛只需跑通参考代码(OpenClaw + ComfyUI 超级英雄照片
  生成 workshop);链接页即讲义网页版,无附加要求。节点 bundle
  (`~/build_a_claw_workshop-bundle`,模型 ~53GB)完好。
- **执行**:讲义打两处补丁(bind lan→loopback + 移除
  `dangerouslyDisableDeviceAuth`;1.3 超时 60→300s),ctl 脚本 0.0.0.0→127.0.0.1、
  openclaw-ctl 还原纯净版(剥离 07-12 StepFun env 注入),`jupyter nbconvert
  --execute` 无头跑完全部章节,**0 单元格错误**;chat UI 经 SSH 隧道 + 设备配对
  完成 5.3 终测,Agent 调 skill 回图成功。证据三件套(执行版 notebook + 双路径
  生成图)+ 偏差清单落 `results/basic_completion/`。
- **事故与教训**:qwen3.6:35b 运行态 34.5GB 与 ComfyUI 同驻,系统可用内存一度
  4.9GB,连带 OOM 了正在跑的 reid-final(未写出产出,旧档无损);停 ComfyUI 后
  原命令重启。**教训:workshop/演示类大模型任务与主链任务必须错峰。**
- **遗留**:公网活演示(多节点版 9000/7072 端口方案)与安全纪律冲突,默认不开,
  需用户拍板;`~/.config/stepfun/stepfun.env` 仍存 API key(违反凭据纪律,
  AutoTune 已关账,建议尽快撤销/清除)。

### 增量 D6-6:自动空间生产器替换人工五区(下午)

- OOM 恢复后先过安全和内存门:`scripts/spark_healthcheck.sh` 输出
  `✅ SPARK CLEAN`;121 GiB 总内存、70 GiB available、0 swap,无 ComfyUI、Qwen、
  ReID 或旧空间分类进程。本增量没有再次启动 ReID,只复用已落盘的小型产物与
  Spark 上现有 Nemotron 12B 视觉服务。
- 自动候选生产将 2278 条空间观测整理为 168 个 track,每轨最多 3 个独立目标裁剪;
  504/504 个视图返回合法结果、失败 0。5 个低信息 track 保留诊断但业务票清零,
  最终得到 53 个自动视觉实例;候选 normalized hash 为
  `0ced8a84133c0862682dcea0dcd2905434dbf68d1241209a13ce16e6959518e2`。
- 可信门将 detector 支撑数与 VLM 语义票拆开:`semantic_observation_count` 只计独立
  目标视图,raw `observation_count` 只贡献 10% 有界持续支撑;小于 2 个 20,000 像素
  有效视图的候选不可参与分配。runner-up 按物理实例计算,同实例 sibling 候选不会
  制造假 0 margin。
- 生产 anchor contract 只冻结 anchor/support/capacity 要求,可拒绝但不可回填模型
  输出。严格分配得到 5/5、`gate_status=PASS`、`needs_user=0`,全局 margin
  `0.06333333 ≥ 0.05`;assignment hash 为
  `43958ebfedeb50afb344bf551a869da614d923bee72161d9530b78a0361a6777`,空间 normalized
  hash 为 `37ace5cb7ea7cd5115660765a378f8a081634774c7a2b73994d03314e175d184`。
- 独立 scorer 不含 candidate/track/region ID,只消费生产侧五区并评分;结果 5/5、
  精确语义匹配 5、额外预测 0、support/capacity mismatch 均 0,normalized hash
  `3f4f38e330acd671267c87aeef90e96e77baa1d91be52c3f5e73fb4cb991d980`。两项电源差异
  仅作信息记录,不扩写为空房安全结论。
- 正式运行 `results/hero/s1-auto-final-v1/` 完成 3306→20 可信投影、4/4 澄清封顶、
  3 个生活组合覆盖 15 件、2 个技术装箱单元承接其余 5 件、5 个 placement 单元、
  `PLAN_READY`、5 张 `REGION_PLANNED` 任务卡。placement 与任务卡均为 20 个唯一
  `hero_*` 实体,缺失/额外/重复均 0;三条风险规则保持 `DEFERRED/non-blocking`。
- 四 Agent 主链回放 `PASS`、`main_chain.complete=1`;bundle 收录 46 个阶段产物并
  逐项通过 SHA-256。生产器没有读取 scorer truth,scorer 没有产出 `regions.json`,
  下游区域 SHA 与生产区域一致。结果页为 `results/hero/s1-auto-final-v1/index.html`。
- 实现链 commits:`c61204ee`(自动 VLM 分配)、`292c543a`(断点运行不误启 ReID)、
  `7897cd69`(生产 contract)、`fe926bbf`(独立目标视图真票)、`c8d30026`(保守语义
  碎片归并)、`8bf22b54`(低信息隔离与物理 runner-up)、`b604742d`(持续观测支撑并
  禁止长时弱相似合并)、`0ce17bd7`(冻结正式主链 51 个文件)。全仓验证
  `290 passed`,`git diff --check` 通过。

### 增量 D6-7:成果页与 110 秒评委主线重构(傍晚)

- 正式入口统一为 `results/hero/s1-auto-final-v1/index.html`,首屏从技术台账改为
  “把旧家的生活组合带到新家”:真实新房帧与五张旧物 crop 直接上屏,四个主数字只保留
  `3306→20 / ≤4 问 / 自动空间 5/5 / 5 张任务卡`。拥挤的十二项导航压为“30 秒主线、
  新家五区、代表任务、完整证据”四项,完整库存、风险、澄清和 SHA 表继续保留下钻。
- 自动空间不再只有计数器。页面从最终 `spatial/assignment.json` 的 `frame:` 引用中
  确定性选择三张已拉回新房 JPEG,以最小帧集覆盖全部五个最终 assignment;每张只显示
  “有几个最终区域引用”,不复用旧 review 的人工结论。生产门 `PASS`、自动接受 5/5、
  独立评分 5/5、待人工区域 0 在同屏对照。
- 代表任务固定展示“学习文具箱→书桌”及七张真实物品 crop,让“识别如何变成动作”在
  一屏内成立;完整五张卡仍在证据区。图片补齐语义 `alt`,键盘焦点、锚点避让、窄屏导航
  与 390 px 响应式规则同时落地。浏览器复核为桌面/390 px 均无横向溢出、破图 0、
  console warning/error 0,主按钮可唯一定位并正确跳到 `#demo-story`。
- 新增 `docs/演示视频脚本_v2.md`:110 秒顺序收敛为“生活问题→可信库存→组合箱单→
  自动空间→任务卡→双生态调优→四 Agent 回放”;旧 v1 明确标为历史底稿,删除旧 run、
  人工视觉阶段、681 消息和结尾失败数字卡等过时口径。
- report 新鲜度补齐此前漏记的 `inventory/clarifications.jsonl`、`group/metrics.json`、
  `spatial/assignment.json` 和 `risk/assessments.json`,避免输入变化后错误复用旧页面。
  正式 index、report state 与 bundle 中的页面 SHA-256 均为
  `a80a6e8783f5a550c71ec2f954fff9253aae8a3b4bcd9397d765dd77e93bb568`。
- 实现提交 `44db3241`;新增失败门保护、真实帧最小覆盖、report freshness、图片替代文本
  与移动端回归测试。全仓验证增至 `292 passed`,`git diff --check` 通过。

### 增量 D6-8:队友可读展示、风险方向退出与剪刀证据纠偏(夜间)

- 成果页面向非技术队友完成第二轮收口。`技术 closure 冻结成员:hair_clip` 改为
  `已确认与「洗漱护理」物品一起打包`;技术装箱单元改为“玩具收纳/零食收纳”,
  `placement 单元`、`PLAN_READY`、生产门、候选实例和四 Agent 状态码分别改为
  收纳组合、布局已生成、区域检查、候选区域和四步自然语言流程。实体、区域、箱号、
  任务卡等内部 ID 不再外显,技术文件与校验码收进默认折叠区。
- 风险提醒方向正式退出比赛展示:冻结配置将 `risk.enabled` 设为 `false`,report 不再
  消费风险产物,当前 bundle 只收录启用阶段;页面、导航和技术复核表均不出现风险区。
  既有风险代码、历史 state 与旧产物保留作研发记录,未删除历史证据。当前 bundle
  为 44 个产物、10 个阶段,阶段集合中无 `risk`。
- “剪刀”错图追到确认数据而非单纯前端文案:`old_1_t2740` 同轨包含发卡与剪刀帧,
  不适合作为可信剪刀成员;仅从 scissors 的确认轨迹中移除该污染轨,保留干净的
  `old_1_t3129` 等证据。正式库存重新投影后仍为 3306→20、4/4 问题封顶、20/20
  下游覆盖;剪刀主图现为 `old_1_t3129_f005580.jpg`,projection hash 为
  `6bd9afe5f5f5ea94ef151f2cdfcf1803d2a2267a0871bf6cbd846badeaca3f5c`。
- `docs/演示视频脚本_v2.md` 同步改为可直接讲述的用语:5 个搬运箱、168 个候选区域、
  5 个最终区域、布局已生成和完整运行记录;主片不再提风险延期表。正式页面经本地
  HTTP 实际返回核验,旧内部术语、风险入口和污染轨均未出现;index、report state 与
  bundle 中页面 SHA-256 一致为
  `d3af7c65633ce8922278c33e73d9548d7677a7205b6c1849b85038ce66494440`。
- 实现提交 `fd5c3542`;新增结果页防外显、历史风险过滤及禁用阶段不进入 bundle 的
  回归门。定向测试 `36 passed`,全仓验证 `294 passed`,`git diff --check` 通过。

### 增量 D6-9:验收双 Agent 并行分支与选卡范围(夜间)

- `EXEC→MEM/SPACE→EXEC` 从同进程内的顺序函数调用改为真实进程边界。EXEC 先生成
  不可变验收请求,随后在等待任一结果前同时启动独立 MEM 与 SPACE worker;两者分别
  只写自己的 trace fragment,EXEC 在角色、覆盖集合、请求 payload hash 与因果引用
  全部吻合后才执行确定性 fan-in。进程 PID、起止时间、重叠时长和退出码另存
  `fanout-run.json`,不混入确定性 trace 哈希。
- 验收链改为失败安全:所选卡缺少相关照片引用、引用文件缺失或为空、任一 worker
  失败/超时、角色结果缺失或越界时,均不生成 combined messages 与 verdict;启动新一轮
  前先清除旧 final outputs,避免失败后误读上次成功结果。Hero pipeline 同时把照片字节、
  worker 代码与两份角色 fragment 纳入新鲜度及交付边界。
- `AcceptanceManifest.selected_card_ids` 建立正式选卡合同:省略或 `null` 表示全卡验收,
  显式列表必须非空、唯一且属于任务卡集合;adjudication 不得越出所选范围。只选代表卡
  时只产生该卡请求与 verdict,未选卡保持原状态,不再被错误改写为 `FAILED`。
- 开发 fixture 增加三张明确标注为 synthetic placeholder 的最小 PPM,仅用于自动化协议
  回归,不构成真实物理执行证据。正式 hero 配置仍保持 verify 关闭,没有作出
  `PHYSICAL_EXECUTION_VERIFIED` 声明;真实复原照片仍是最后的现场证据,不是本轮并行
  架构实现的前置条件。
- 实现提交 `812f0989`。全仓验证 `309 passed`;`compileall` 与
  `git diff --check` 通过。本地完整开发链落在
  `/private/tmp/dgx-verify-fanout.dyQn2o`:MEM/SPACE 实测重叠约 87.53 ms、两进程均
  返回 0;strict trace replay 为 `PASS`,20 条消息、3/3 请求闭合、2/2 adjudication
  闭合。结果分布为 `VERIFIED=1 / NEEDS_USER=1 / FAILED=1`,证明 fan-in 保留独立判断
  而非把所有分支强行包装成成功。

### 增量 D6-11:验收 Agent 迁入 Spark 与代表卡就绪(夜间)

- 正式 verify 执行位置由 Mac 本地阶段迁为 Spark stage:任务卡、验收清单、父 trace
  与所选照片以项目相对路径定向同步到 `spark:~/proj`,总量超过 50 MiB 即拒绝;跨境
  传输失败重试一次,仍失败则停止。远端 EXEC 同时启动 MEM/SPACE,完成后只逐文件拉回
  requests、两份角色 fragment、combined messages、verdict、更新任务卡与
  `fanout-run.json`,不拉取无关 results 大目录。
- 远端 runner 补齐真实退出码 marker。任一 worker 或远端命令非零时立即向本地传播
  退出码和日志,不再把快速失败拖成编排总超时;拉回前删除本地旧声明产物,失败路径不会
  误读上次成功结果。实现提交 `08213054`。
- 从 `08213054b8aec4c424a6e6620e7242bdaa0ffae7` 的独立干净 checkout 通过标准
  `scripts/deploy.sh` 部署,避免工作区并行 ReID 实验进入 Spark。强制安全门两次均为
  `✅ SPARK CLEAN`;实机 `spark-48f0` 合成冒烟中 MEM PID `1216003`、SPACE PID
  `1216004`,重叠 `92.055 ms`,两支退出码均为 0。远端与拉回本地的 requests/MEM/
  SPACE/messages 四份 SHA-256 逐项一致。
- Spark 产出的三卡结果仍为 `VERIFIED=1 / NEEDS_USER=1 / FAILED=1`;本地 strict
  replay 为 `PASS`,20 条消息、3/3 验收请求闭合、2/2 adjudication 闭合。该运行只用
  明确标注的 synthetic fixture 验进程与协议边界,没有加载 Nemotron,也不构成物理执行
  证据。
- 当前正式代表任务收敛为 `card-02 学习文具箱→书桌`;验收模板已加入
  `selected_card_ids=["card-02"]`,七类实体全部预填 `present=false`,照片路径冻结为
  `local-data/hero_s1/acceptance/study_desk_after.jpg`。缺少该真实照片时实测在 worker
  启动前以 `FileNotFoundError` 停止,输出目录无文件。准备提交 `4e6c4725`;正式配置继续
  保持 verify/strict 关闭,照片到位前不作 `PHYSICAL_EXECUTION_VERIFIED` 声明。
- 全量后端验证 `351 passed`;`compileall` 与 `git diff --check` 通过。

### 增量 D6-12:参赛技术介绍精简定稿(夜间)

- 参赛技术稿重写为 `docs/参赛项目技术介绍.md`,31 行、约 1800 字,用一篇连续文章和
  一张紧凑技术栈表说明选题、四 Agent 实现路径、Spark 本地模型服务与可回放证据链;
  实现提交 `56694ee0`。
- 结果数字逐项回查正式产物:库存 `3306→20`、澄清 `4/4`、组合覆盖 `20/20`、空间
  `2278→168→5`、独立评分 `5/5`、主链 replay `PASS`、bundle 44 份产物,文稿口径与
  `results/hero/s1-auto-final-v1/` 一致。
- TensorRT 仅表述为视觉推理部署链路正在进行引擎构建、输出一致性与吞吐测试,待并行
  实现产出证据后再写入结果数字。正文未扩写研发缺口,排版检查与 Markdown 空白检查通过。

### 增量 D6-13:队友展示固态单文件(夜间)

- 当前正式成果页冻结为
  `results/hero/s1-auto-final-v1/AI搬家复原_队友展示_单文件.html`:页面样式、
  文字和 23 个独立图片素材全部内嵌,56 处图片引用均转为 `data:` URI,
  不再依赖 `results/` 目录结构、本地服务或网络资源。
- 单文件大小 3,396,227 bytes,SHA-256 为
  `119bbe7922e43b27a9f541b19e095ade591bb29639bb1c1adaa12dabe4f74505`;源页 SHA-256 为
  `ca6c5736b2070e3a03083309275fe73b7a675360921f2e26a76dd7d23acf4f1b`。
- 归一化页面结构与源页完全一致;56 个 base64 载荷全部通过严格解码与 JPEG
  文件头校验,外部资源依赖计数为 0。自动化浏览器按安全策略禁止直接访问
  `file://`,未绕过限制,改用结构一致性和内嵌资产解码完成离线验证。

## 失败与教训

- AutoTune-v1 将澄清从 1379 降至 677,但完整合并仍为 14/20、R@1 仅 0.8083,
  未达 16/20 与 0.90 的预注册门。代理指标能选工作点,不能替代冻结人工 GT。
- R2 Worker 方案在 Spark 网络现实中被 DNS/SNI 阻断证伪;改为限时 S3 预签名 URL
  后才满足无节点凭据、可恢复和可清理要求。传输链必须用真实双向 E2E,不能只测 Mac。
- Qwen 35B 与 ComfyUI 同驻造成 OOM 并拖挂 ReID。无 swap 的统一内存环境必须按
  工作负载错峰,每次模型任务前检查 `free -h`,OOM 后先自检和查残留进程。
- 自动空间一版曾允许相隔 67 秒、DINO cosine≈0.63 的同语义轨合并;两件同类家具
  也可能满足该条件。门审在正式运行前删除长时弱相似分支,改用 10% 有界持续观测
  支撑区分候选,两个视觉实例始终保持独立。
- 原结果页把口径、风险和哈希放在价值主线之前,且旧演示脚本仍指向人工视觉阶段与
  `s1-final`;证据虽然完整,评委需要滚动多屏才能找到 3306→20、自动五区和任务卡。
  展示入口必须和正式 run 一起冻结,不能让录屏人员临场辨认多个旧目录。
- 当前 5 张任务卡的 `priority` 全为 3。经本轮范围复核,优先开箱不影响比赛技术闭环
  与核心价值叙事,从“唯一核心缺口”降为可选产品增强,不再挤占展示优化和彩排时间。
  二维码同样保留为可选项;复原前后照片和现场安全事实继续按范围延期。
- “剪刀”错图说明可信投影仍需检查证据轨是否内部一致;一个轨迹 ID 可以同时含正确帧
  和污染帧,不能因类别名正确就直接冻结。展示层也不应暴露内部枚举、英文 ID 和状态码;
  可审计性应放在折叠证据区,不应牺牲主叙事可读性。
- 旧验收实现虽然在 trace 中使用 MEM/SPACE 角色名,实际仍由 EXEC 进程顺序调用同一
  模块,只能证明协议形状,不能证明独立 fan-out;照片字段也只校验字符串,选一张代表卡
  时还会把未选卡判失败。独立性必须由进程边界、各自 fragment、时间重叠与失败传播
  共同证明,照片存在性与验收范围则必须在启动 worker 前失败安全地冻结。
- 共享工作区的另一任务在暂存检查与提交之间写入同一个 Git index,第一次提交意外夹带
  并行 ReID 文件。随即只移动本地提交/index 指针、不改工作区文件,完整恢复并行改动,
  后续两次提交改用隔离临时 index。多任务共享 worktree 时,“提交前看过 staged”仍不足;
  提交本身也必须隔离 index 或在提交后立即核对文件清单。
- 参赛技术稿首版过长,技术台账感压过项目叙事,且 TensorRT 尚在并行测试时不宜提前写成
  完成态。先以 `dd98dfec`、`6f6ee2e7` 回退两次文档提交,再用实测事实重写短稿;参赛稿
  应让实现路径和结果自然形成亮点,状态口径必须晚于证据。

## 明日计划

- 按 `docs/演示视频脚本_v2.md` 完成一次 110 秒桌面彩排,只录正式入口;检查四段主线、
  三张新房证据、代表任务与四 Agent 回放在不滚长表时能连续讲清。
- 冻结录屏素材清单与镜头时码,把 Nemotron 15×、StepFun 调优代理和运行时零云依赖
  压在 14 秒内,不让模型日志盖过用户价值和可执行任务。
- 演示前用冻结配置做一次内容寻址复跑,确认自动空间无需人工 region/anchor ID 即可
  重现 5/5、20/20 和 trace `PASS`;ComfyUI、Qwen 35B 与主链模型继续严格错峰。
- 彩排时由一位未参与实现的队友只看成果页复述四步主线;若仍需解释内部名词,继续改写
  展示文案,但风险提醒方向不再返回正式页面或当前交付包。
- 优先开箱和二维码均留在可选增强池,仅在不影响彩排与成片时处理;真实复原照片到位后
  直接用已冻结的 `card-02` 模板打开 verify/strict,复核 Spark-local PID 重叠、四份
  trace 哈希和最终 bundle,再把完整往返 trace 纳入正式演示;在此之前不作物理执行已
  验证声明。
- AutoTune 已关账后撤销并清理 Spark 上遗留的 StepFun API key,不把凭据继续留在
  曾被入侵的 `Developer` 账户。
- TensorRT 并行实现形成引擎、输出一致性和吞吐证据后,只替换技术稿中的测试态一句并
  补入对应实测数字,保持正文篇幅与现有叙事结构不扩张。
