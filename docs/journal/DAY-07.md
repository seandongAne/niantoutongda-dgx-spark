# DAY-07 · 2026-07-19

> 赛程日:D7
>
> 当日主责:GDINO 运行时等价复核定案与三线性能优化执行(BF16 选择性混合 / TRT 哨兵二分 / Torch-TensorRT 文本外置)
>
> 状态:完成(运行时等价三线收官:口径合同落地,BF16 扩集回归 5/32 否决关线;TRT 根因定案=融合依赖漂移,修复配方使 FP32 引擎首过严格门 629.96ms、FP16 修复版 2.49× 消除灾难但贴阈值关线;文本外置编译四连成而运行挂死,60 分钟死线止损;入账优化维持 compile FP32 1.170×;技术稿测试态句按"过门才更新"规则以实测数字替换)

## 今日目标

- 在不再读取两次终检 GT 的前提下,试验 schema 转写与实例级语义签名重排;先分离
  选参/冻结证据边界,再测 PyTorch AMP、Torch-TensorRT 混合执行和 TF-TRT 前置门。
- 复核 Grounding DINO 的 precision policy、encoder proposal、TopK query 排位与
  ONNX 后处理集合等价;同时测 `torch.compile`、模块耗时构成和 Torch-TensorRT
  真正 engine build,修正只看 raw query-slot 最大偏差的旧口径。
- 依据复核结论执行三线优化(用户已批准,优先级①→②→③,GPU 任务严格串行):
  ① BF16 选择性混合精度——先做 compile BF16 丢失检测的 margin 分析,再让视觉
  backbone+encoder(合计 ~90% 耗时)走 BF16 autocast、decoder 与对比头留 FP32,
  过同一冻结检测集合门;② 用与原图最终输出 bit-exact 的插桩 ONNX 构建 TRT FP32
  no-TF32 engine,在哨兵边界(proposal scores/coord logits/TopK 索引/final)与已
  验证的 ORT 对比,把现有 TensorRT engine 的检测丢失二分到第一个越界区段;
  ③ Torch-TensorRT 把文本 backbone(0.56% 耗时)移出 torch.export 边界,eager
  预计算 text features 作为显式输入,绕开 BERT embedding `aten.add` 符号 shape
  构建失败,新 dry-run 证明回退真实生效后才谈 engine 与速度。
- 新冻结合同把 matmul/cuDNN TF32、float32 matmul precision、模型、processor、
  输入与集合匹配器哈希一起封存,不再依赖框架默认值。

## 关键证据或截图

### 增量 D7-1:语义重排、AMP 与 Torch-TensorRT 混合门(凌晨)

- 实验在隔离分支 `codex/semantic-perf-20260718` 完成,实现提交 `fdead9a5`,证据提交
  `e9159a3a`。转写层新增中英文 display/alias/compound 对齐、置信度与审计 evidence;
  多来源冲突和歧义复合标签一律 fail closed。伪标签生成允许完全省略 GT,限定
  candidate-only tutor 来源并冻结 stitch/link/ingest/tutor 哈希,已有证据拒绝覆盖。
- 实例级语义签名将转写 identity 与 S5 可比属性组合,只重排既有视觉 Top-5 候选。
  未读冻结人工 GT 的诊断实验得到 116 个 pseudo identity/502 个 tracklet;hash split
  为 dev 74 identity/340 tracklet、proxy partition 42/162,tracklet 交集 0。语义权重
  `0.15` 在 127 个 dev query 上使 R@1 `0.4961→0.6378`;已打开的 58-query partition
  为 `46→51` 个 Top-1 命中,点估计 `0.7931→0.8793`。
- 上述语义数字不构成独立冻结集结论:既有 SF1 projection 的训练标签覆盖本轮
  502/502 tracklet,旧 selection 还输出过全体转写覆盖率;两侧 Wilson 95% 区间分别
  `[0.6723,0.8775]`、`[0.7712,0.9403]`。因此只保留为 transductive diagnostic,
  不写成 hero R@1 `0.8083→0.8793`;该 partition 已消耗,转写器修订后不得重开。
  新评估器默认拒绝 learned projection,select 只构建 dev signature,holdout 才构建
  holdout signature,并把 evaluator 与全部数据输入哈希一起冻结。
- PyTorch forward-only AMP 在同一 batch 2 冻结输入、100 次测量下:FP32 P50
  `680.43 ms`;FP16 `497.98 ms`、`1.366×`;BF16 `502.50 ms`、`1.354×`。FP16
  检测数仍为 `[5,5]`但标签/顺序结构不等价;BF16 检测数变为 `[4,5]`;两者严格
  决策门均 FAIL。显存 allocator 峰值仅少约 40 MiB,不据此声明统一内存显著下降。
- 39 个阶段、每张量 4096 点的漂移探针显示误差分布在文本、视觉、encoder/decoder:
  FP16 首个 `>1e-3` 观测在 BERT layer 1、首个 `>0.1` 在 Swin stage 1;BF16
  分别在 BERT layer 0 与 text projection。阶段采样不能证明单一因果算子,因此
  自定义 converter 维持 NO-GO,等待 FX/ATen 算子级 trace 与 selective FP32 island。
- Torch-TensorRT 使用 digest 固定的 NVIDIA 26.06 基础镜像,隔离派生层只安装
  `transformers==5.13.1`;运行栈为 PyTorch/Torch-TensorRT `2.13.0a0`、TensorRT
  `11.0.0.114`、CUDA `13.3`。提交版 v2 在第一道容器 eager FP32 门停止:P50
  `657.45 ms`,但对现有 PyTorch `2.13.0+cu130` 冻结 oracle 的有限 logits 最大偏差
  `4.1671`、boxes `0.9841`;verdict 为
  `NO_GO_CONTAINER_EAGER_FP32_BASELINE_MISMATCH`。未进入 TRT 分区/编译或 FP16,
  也未把容器自身延迟写成加速。
- TF-TRT 前置审查为 NO-GO:项目环境无 TensorFlow、无 native TensorFlow
  Grounding DINO SavedModel,现有 ONNX oracle 也未对齐 PyTorch。未拉 TensorFlow
  容器、未产生不可比性能数字。总对比和机器可读结论落在
  `results/acceptance/SF1/optimization-comparison-20260718/`;全量后端验证
  `360 passed`,`py_compile` 与 `git diff --check` 通过。

### 增量 D7-2:TopK/TF32 复核、compile 与混合 TRT 实际构建(凌晨)

- Spark 安全门保持 `✅ SPARK CLEAN`;每次模型加载前 `free -h` 均为约 70 GiB
  available、swap 0,既有 Nemotron vLLM 未停止。全部长任务用 nohup 与独立日志,
  远端证据先逐文件拉回;代码提交为 `8d56e69b`、模块回退参数提交为 `7c20d3d5`。
- 旧冻结 oracle 未记录 precision policy。当前 host 默认状态为 matmul TF32
  `false`、cuDNN TF32 `true`、float32 matmul precision `highest`;显式复现该状态时
  与旧 oracle logits/boxes bit-exact。切为双 TF32 `false` 后,两批 TopK 集合仍各
  900/900 相同,但同 rank 仅 837/900、840/900;raw boxes 最大偏差 `0.98414`,按
  encoder proposal ID 对齐后约 `2.0e-4`,检测集合严格 PASS。
- NGC 与 host 同为双 TF32 `false` 时,TopK 只剩 4/0 个槽位变化且集合完全相同,
  检测集合严格 PASS。两端各用框架默认 cuDNN TF32 时,不同 cuDNN build 使同 rank
  降到约 223/224,并替换 4/3 个 proposal。旧 v2 的大 raw 差异因此包含 precision
  policy 与 query 排位混杂,不能直接判定容器检测语义失败。
- ONNX 原图与插桩图最终输出 bit-exact;图中确认一个 `K=900` 的 TopK 和 48 个
  GridSample。相对 host no-TF32,proposal score/coord 最大偏差分别 `7.44e-5`、
  `2.11e-4`;第二批 TopK 仅替换 1/900 proposal。正式 AutoProcessor 后处理为
  `[5,5]→[5,5]`,10 个检测全部完成标签约束最大 IoU 匹配,最差 IoU `0.9999983`,
  score 差 `2.74e-4`,框差 `0.00025 px`,strict diagnostic PASS。相对旧默认 oracle
  也为 `[5,5]→[5,5]`,最差 IoU `0.9999797`;因此原 `4.17/0.984` ORT raw 剪刀差
  中的槽位与 boxes 部分主要来自非决策 query 错位,不是此前 TensorRT `[1,3]` 的
  检测退化;proposal-ID 对齐后 final logits 最大偏差仍为 `1.3666`,不能把全部漂移
  归因于 TopK。
- `torch.compile` FP32 将 P50 从 `680.78` 降至 `581.83 ms`,为 `1.170×`;检测数
  `[5,5]`,score 差 `≤7.79e-5`、框差 `≤0.00037 px`,门禁 PASS。首次 lazy
  invocation 为 `59.1 s`,仍有 10 个 `Tensor.item()` graph break。compile BF16
  P50 `312.84 ms`、`2.176×`,但检测数 `[4,5]`,正确性 FAIL;不启用。
- 模块级 FP32 profile 经 unhooked frozen-oracle 与 hooked bit-exact 双门:视觉
  backbone `214.39 ms/31.36%`,encoder `400.95 ms/58.65%`,decoder
  `49.08 ms/7.18%`,文本 backbone `3.83 ms/0.56%`,其余 `14.94 ms/2.18%`。
  视觉 backbone 即便单独 3×,Amdahl 全链上限也只有约 `1.264×`,未扣分区开销。
- Torch-TensorRT v4 的容器 eager no-TF32 检测集合先通过,export 兼容改写与
  `torch.export` 对 same-process eager 均 bit-exact。dry-run 为 4520/4521
  converter-supported、规划两段 TRT partition(3280+1240 算子),99.98% 覆盖;
  实际 FP32 build 在 BERT `embeddings + position_embeddings` 的
  `aten.add.Tensor` converter
  触发 `ValueError: __len__() should return >= 0`。experimental decompositions 未
  改变失败点;指定 module FQN 的回退请求也未生效,原因尚未定位。没有形成 engine,
  因此没有 FP32/FP16 混合延迟或加速数字。
- 复核版报告与机器可读摘要落在
  `results/acceptance/SF1/optimization-comparison-20260718/`;正式边界对比位于
  `onnx-pytorch-boundary-comparison-20260718/`,混合构建证据位于 v4-v6 目录。
  证据提交为 `8e3d11f6`;全量后端复跑 `360 passed`,全部新脚本 `py_compile`、
  JSON 解析和 `git diff --check` 通过。

### 增量 D7-3:三线执行——margin 定案、选择性 BF16、TRT 哨兵定位与文本外置编译(晚间)

- 发射前安全门 `✅ SPARK CLEAN`,首连 70 GiB available、swap 0;全部长任务
  setsid -f 后台+独立日志,GPU 阶段严格串行(opt_chain/chain2/v8/v9 四次发射)。
  隔离分支在 `/private/tmp` 临时检出中的 7 个提交(fdead9a5..ffeb3fd8)已 fetch 为
  `backup/semantic-perf-20260718` 并 `--ff-only` 收编 main,消除重启即失的证据风险。
- **margin 定案**:compile BF16 `[4,5]` 丢失的检测=img0 q2,FP32 分数 `0.2300`
  对阈值 `0.22` 边距仅 `0.010`,BF16 推至 `0.2173`;img0 另有两检测挤在
  `0.2217-0.2263`。BF16 类候选在分数容差 1e-3 的 strict 门下数学上不可过,
  可判别口径为 decision-set strict/diagnostic 双档。
- **线① 选择性 autocast**(`gdino_selective_autocast_bench.py`;双 TF32 false +
  matmul precision highest 显式封存;region 钩子进出 autocast 并将区域输出回铸
  FP32;fp32_repeat 位精确):三候选全部 `[5,5]`、10/10 一对一配对、零翻转——
  **BF16 检测丢失被选择性精度消除**。encoder-only 将配对分数差压至 `≤3.4e-3`;
  当前唯一挡 diagnostic 门(IoU≥0.99/分数≤1e-2)的是 img1 单框 `3.9-8.3px` 漂移
  (IoU `0.977-0.982`),源头为 backbone/encoder BF16 扰动 proposal 初始框,FP32
  decoder 精修未完全收敛。速度(eager,对同策略 fp32 681ms):backbone `1.136×`、
  encoder `1.152×`、双区 `1.336×`。verdict=
  `NO_GO_ALL_SELECTIVE_CANDIDATES_FAIL_SET_GATES`,按冻结口径如实记 FAIL;若未来
  裁决"集合等价+IoU≥0.98"口径,双区 1.336× 即刻可用——口径裁决权在用户。
- **线② TRT 哨兵定位**(`gdino_trt_instrumented_sentinel_dump.py`):bit-exact
  插桩 ONNX 经 trtexec `--noTF32` 构建诊断 engine(60s/952MiB),12 哨兵对 ORT
  参照。verdict=`DIVERGES_BEFORE_TOPK`,实际收窄远超区段级:proposal 坐标
  `2.1e-4`、encoder GridSample `1.5e-4/4.6e-5`、proposal-ID 对齐后最终框 `1e-5`
  全部干净;**只有类别 logits 坏**(encoder 对比头 max `4.8`/mean `0.61`,final
  logits max `5.9`),TopK 集合因此仅 613/484(每批 900)重合、检测 `[2,3]`。
  GridSample 在 encoder 正式排除;视觉记忆、文本入融合、decoder 框路径全部无辜。
  **TRT bug 收窄至文本-图像对比头 matmul 或其 slice_scatter 填充二选一**;下一层
  二分只需在 scatter 前加一哨兵;若坐实 slice_scatter,可用 concat 填充改写修复。
  插桩 engine 阻断融合,计时不可比,不产生速度数字。
- **线③ 文本外置**(`--text-outside-export`:BERT last_hidden_state 由 forward
  hook 捕获后 eager 预计算,作第 6 个显式导出输入;stub 位精确门/占位符计数门/
  参数残留门三重 fail-closed):v7/v9 两次证明——图 4521→4225 算子(BERT 296 个
  移出)、占位符 6/6、`text_backbone` 参数零残留、stub 对 patched eager 位精确、
  dry-run 99.98% 两段 TRT 分区(2984+1240),**真实 FP32 engine 编译两次成功,
  卡三轮的 `aten.add` 转换器失败点被绕死**。但 profile/benchmark 阶段两次 OOM
  (exit 137,进程峰值约 +36GiB 撞 Nemotron vLLM 51GiB 驻留),`probe.json` 因
  SIGKILL 未落盘,证据=partition/compiled_graph 侧文件+日志。v8 另证:
  `offload_module_to_cpu` 与 cuda 示例输入发生 FakeTensor 设备冲突,不可用。
- Commits:工装 `4734724e`/`e7ee723e`/`f06b38e4`/`34474b65`,证据 `1bbbc91d`/
  `8e70b098`/`9a64e458`/`d9976001`,journal 手术 `58a93562`。

### 增量 D7-4:口径修订、BF16 扩集回归否决、二分收窄与 v11 运行期死刑(深夜—次日晨)

- **口径修订落成合同**(`docs/运行时等价判定口径_2026-07-19.md`,提交 `9ef953b7`):
  用户裁决新增第三档 revised-0.98(集合等价+一对一标签匹配+IoU≥0.98+分数差
  ≤1e-2),strict/diagnostic 两档原样保留。合同同时载明采纳前置:冻结对 PASS、
  ≥24 帧未参与调参的扩集回归、主链集成回归,三关全过才准进主链。**并修正我此前
  的口头误报**:双区 BF16 实测 img1 min IoU `0.97736`,在新口径下同样 FAIL,
  "双区 1.336× 即刻可用"不成立;冻结对上只有 encoder-only(`0.98197`,`1.152×`)
  过 revised-0.98——口径不迁就候选。
- **线① BF16 扩集回归否决**(`gdino_selective_bf16_regression.py`,32 帧确定性
  均匀抽样,evidence 提交 `84319689`):verdict=`REVISED_098_FAIL_5_OF_32`。
  5 个失败帧全部是**一对一匹配不完整**(检测数 2→1、3→4 与两例 5→5 标签翻转),
  IoU/分数本身反而都在门内——贴阈值翻转在开放帧上是常态而非冻结对特例。
  encoder-only BF16 未过采纳前置②,**BF16 线关闭**;已入账运行时优化维持
  `torch.compile` FP32 `1.170×` 唯一。
- **线② 二分收窄两级**:v1 教训=`make_empty_tensor_value_info` 输出无 dtype,
  TRT parser 拒收 UNKNOWN(0);v2 回填 dtype 后拿到判决
  `UPSTREAM_OF_SCATTER_GUILTY:val_8892`——图里名为 `slice_scatter` 的节点实为
  **Transpose 纯搬运**(输入 `val_8892 [256,20906,2]` 与输出脏度逐位相同,max
  `4.818`),scatter/填充路径洗清,罪在更上游。v3(提交 `41a6f274`)穿透
  {Transpose,Reshape,Identity,Cast,Squeeze,Unsqueeze,Flatten} 上溯两层真实算子:
  真实生产链=`matmul_12`(对比头 matmul,[2,20906,12])→ Where 掩码 → ScatterND
  组装 → Transpose,四值脏度逐位相同(max `4.818`)——第一脏值锁定 `matmul_12`,
  判决 `DIRTY_BEYOND_DEPTH2`(其直接输入未入观测)。
- **线② 终局:v4 反转定案"融合依赖漂移"**。给二分器加 `--target-value`(提交
  `2a48cba9`)指向 `matmul_12` 本体:其输出变**干净**(`1.9e-4`),全部上游
  (layer_norm/linear/add)`≤9.2e-4` 全净——同一张量、同一 `--noTF32`,仅因标记
  的观测输出集不同,就在 `4.818` 与 `1.9e-4` 间翻转;而 v4 引擎的最终 logits 仍脏
  (max `2.8`)。合并 v3/v4 与哨兵证据:**逐算子数学全部无辜,脏度总是"逃离"被
  materialize 的区域——根因是 TensorRT 对对比头区域(编码器侧与解码器侧两个同构
  text-image matmul 子图)的融合 kernel 编译缺陷,谁留在融合里谁算错**。这是可
  最小化上报 NVIDIA 的 bug 模式;工程侧对应的修复假设=在生产引擎上强制拆分该区
  融合(附加哑输出/图外科),需重过检测门与计时才有资格谈采纳。二分证据:
  `trt-scatter-bisect-20260719-{v2,v3,v4}/result.json`(timing cache 使 v3/v4
  单轮仅约 90 秒)。
- **线③ v10/v11 两项终局**。其一,OOM 真凶定案:v10 在 vLLM 已停、115GiB
  空闲下仍 exit 137,击杀点=`_profile_trt_events` 用 CUDA/CUPTI profiler 包住
  编译模块**首次执行**;profiler 自述仅是"佐证、永不作门",故加
  `--skip-runtime-profile` 旁路(提交 `a49c236a`)。其二,v11(错峰,跳过
  profiler)FP32 engine **第四次编译成功**(23:14,4225 算子 4224 进 TRT、仅
  `aten.all.dim` 回退,两段 engine 2984+1240),但编译后执行阶段 CPU 99.8%/GPU
  96% 空转 **4.5 小时零产出**——两段式结构正常前向应为秒级,判定运行期挂死;
  按用户批准的 60 分钟死线规则 `docker kill`,链尾 trap 自动 `VLLM_RESTARTED`、
  `CHAIN4_DONE`(03:47Z),错峰窗口关闭。**文本外置最终判定:编译可行
  (aten.add 绕死,四连成),运行不可用;probe.json 从未落盘,技术稿测试态
  表述按"过门才更新"规则不动。**

### 增量 D7-5:融合拆分修复实验(用户批准时间盒)——首个过严格门的 TRT 引擎(晨)

- **构型**:`gdino_fusion_split_repair_prep.py` 把 bisect v3∪v4 的 12 个观测值
  (剔 bool 掩码)以带类型输出加进生产 ONNX(dtype/shape 从 bisect result.json
  回填,v1 UNKNOWN dtype 教训),强制 TRT 在对比头区按已证干净的边界拆分融合;
  消费方合同不变(仍只读 logits/pred_boxes)。工装提交 `b9f91648`/`5c129ae0`。
- **v5 FP32 no-TF32:严格门首过**。决策 `[5,5]/[5,5]`、标签结构相同、分数差
  `2.45e-4`、框差 `0.0027px`——**TRT 引擎第一次通过 D6 冻结严格决策门**
  (此前 FP16/FP32-TF32/FP32-noTF32 全灭)。p50 `629.96ms`:比 D6 无标记
  no-TF32 引擎(647.9ms)**还快**,拆分未付出时延代价;对 eager FP32
  (680.78ms)=1.081×。raw logits max `4.166` 精确复现 ORT 的已知导出漂移形态
  (槽位重排,非数值病)——**TRT 至此忠实于 ONNX,融合缺陷被修复配方消除**。
  生产口径不变:compile FP32 1.170×(581.83ms)仍更快,修复版的价值=正确性
  证明+NVIDIA 上报配方,不是即刻替换。
- **v6 FP16(+FP32 标记精度岛):速度全保、灾难消除、撞 BF16 同墙**。p50
  `273.15ms`=**2.49×**(与 D6 损坏版 277.2ms 持平);检测 `[1,3]→[2,4]`,
  配对框 IoU `0.995/0.974`、框差像素级——融合性乱码已不在。但配对分数漂移
  `0.019-0.024`:同时超 revised-0.98 档的 1e-2 分数上限,且远大于贴阈值检测的
  边距(`0.0017-0.010`),丢失的 4 个检测全是被 FP16 特征噪声推过 0.22 阈值——
  与扩集回归判死 BF16 线的**同一结构性问题**,不是引擎 bug,标记加密度也救不了。
  集合门三档全 FAIL,如实关线。证据:`trt-fusion-split-20260719-{v5,v6-fp16}/`
  (`d12e20e5`/`010a45a0`),新集合门工装 `gdino_npz_decision_set_gate.py`。
- **时间盒交付物**:①修复配方(12 个带类型标记输出,亦是最小上报包的核心);
  ②首个正确性达标的 TRT 引擎(FP32,1.081×);③半精度问题定性收束——从
  "TRT 不可用"降为"与 torch 半精度同款的贴阈值翻转",若未来产品层解决贴阈值
  问题(如滞回阈值/边距感知接受),2.49× 立即在架上。可选残项(未做,收益
  存疑):标记集最小化冲速(追 compile 的 581.83ms,预计追不上)。

## 失败与教训

- identity/tracklet 零交集仍不足以证明模型层独立:上游 learned projection 若已经见过
  holdout identity,或 select 暴露过 holdout 聚合特征,只能称 metric-sealed/transductive
  诊断。冻结合同必须同时覆盖数据、代码、预训练来源与实际读取时序;当前 proxy partition
  已被打开,后续算法修订只能等新 `holdout_b`,不能换名字后复用。
- FP16 检测数保持 `[5,5]`不等于输出等价;标签、顺序或框关联变化同样会破坏下游。
  阶段 hook 的“首次观测漂移”也不是致因算子。自定义转换前应先做标签+IoU 匹配、
  算子级 trace 和 FP32 island 消融,不能从一个 stage 名直接推导 converter。
- Torch-TensorRT 容器包含目标编译器并不保证可与现有 PyTorch oracle 比较;26.06 的
  `2.13.0a0/CUDA 13.3` 与现有 `2.13.0+cu130` 在同权重同输入下已发生大幅有限值漂移。
  必须把容器 eager FP32 放在编译前,否则很容易把运行栈漂移误归因给 TensorRT。
- 上一增量把 query-slot raw 最大差直接当作检测语义门,并把旧 oracle 的框架默认
  精度当成已知条件,结论过严。复核后应拆成三层:shape/dtype/非有限值安全门、raw
  排位诊断、标签+IoU 检测集合门;precision policy 必须显式封存。ORT 在当前样本的
  集合门通过,现有 TensorRT engine 的检测丢失仍成立,两者不能混写。
- Torch-TensorRT 99.98% dry-run converter 覆盖不等于 engine 可构建;本轮实际失败
  发生在已标为 supported 的 `aten.add.Tensor`。experimental decomposition 与未命中
  的 module FQN 回退都不能被写成有效优化。下一步先做最小复现或匹配稳定版本,不在
  未形成 engine 时继续累计名义覆盖率。

- 贴边检测(边距 0.010)让任何数值扰动都可能翻转决策;固定阈值上的检测数门天然
  放大 1e-2 级漂移。BF16 候选的生死应由 decision-set 双档口径给出,positional
  1e-3 门对其无判别力,只作诊断留档。
- 哨兵定位的解释边界:插桩输出阻断 TRT 融合,该 engine 时延与生产 engine 不可比;
  "对比头/slice_scatter"是二选一嫌疑而非已证因果,需 scatter 前哨兵完成最后一层
  二分后才能写根因。
- 共享节点上"编译后执行"阶段的内存峰值(约 +36GiB)连续两次击杀探针;
  `empty_cache` 不足以对抗 51GiB 驻留服务,错峰纪律对编译类任务同样适用。
  `offload_module_to_cpu` 会把权重搬 CPU 而示例输入在 cuda,FakeTensor 设备传播
  直接失败——不是可用的省内存手段。
- **上条归因修正**:v10 证明 OOM 与 vLLM 驻留无关——真凶是 profiler 用 CUPTI
  包住编译模块首次执行,115GiB 空闲照样 exit 137。"佐证型"测量必须天生可旁路,
  否则它就是门。
- 编译成功≠运行可用:v11 两段式混编(理论前向秒级)在执行阶段满载空转 4.5 小时
  零产出。探针只在收尾打印一次 JSON,使挂死与慢基准外部完全不可分;长探针必须
  每阶段/每 N 次 run 打一行心跳,沉默才能成为诊断信号。容器默认无 SYS_PTRACE、
  宿主无免密 sudo,运行中栈不可观测——可观测性要在发射前设计,不能事后补。
- 图中值名不等于算子语义:名为 `slice_scatter` 的节点实为 Transpose 搬运工。
  二分工具必须穿透搬运算子上溯真实生产者,否则定罪永远停在无辜的中转节点。
- 单次构建的二分定罪对 TRT 不充分:标记观测点本身改变融合决策,同一张量在两次
  构建间 `4.818↔1.9e-4` 翻转。"输入净/输出脏"判据只在融合边界固定时有效;结论
  必须来自成对构建的差分(v3×v4),否则会把融合缺陷误写成某个算子的数学错误。

## 明日计划

- ~~错峰停 vLLM 完成 v10/v11~~ 已执行完毕:v11 编译成又挂死,已击杀并恢复
  vLLM 服务,TRT 混编线止损(详 D7-4);若重启该线,先给探针加心跳再谈。
- ~~TRT 根因最后一层二分~~ 已定案=融合依赖漂移(D7-4);~~融合拆分实验~~
  已执行并收官(D7-5):FP32 修复版首过严格门(629.96ms/1.081×),FP16 修复版
  2.49× 但贴阈值翻转关线。剩余可选:NVIDIA 最小上报包(修复配方已是核心素材);
  标记集最小化冲速(预计追不上 compile,不建议)。
- ~~选择性 BF16 口径裁决~~ 已裁决并执行:revised-0.98 口径入册,扩集回归
  5/32 FAIL,BF16 线关闭(D7-4);再启需先解决贴阈值翻转的结构性问题,不是调参。
- 三线执行结果已落档;任一候选过冻结检测集合门,才进入更大独立冻结集回归与
  端到端吞吐;未过门的只记诊断,不产生性能声明。
- 为语义重排采集并冻结未参与本轮选参的新 `holdout_b`;若继续使用 learned projection,
  只准用 dev identity 训练后再封存 holdout。新集到位前只修评估工装,不再产生提升声明。
- 若文本外置后 Torch-TensorRT 仍有构建故障,为 `aten.add` 符号 shape 错误形成最小
  复现并考虑匹配的稳定版本;FP32 engine 集合门 PASS 后才测 FP16/BF16。TF-TRT 继续
  旁路,没有 native TF FP32 SavedModel 对齐时不创建性能结果。
