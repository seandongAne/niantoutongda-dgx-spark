# 验收照片指引 · hero-s1 可选搬后执行复核

> 目标（可选）:为当前正式代表任务卡补充搬后外部证据，走完
> presence ∧ compliance → VERIFIED 全消息链。G7b 已于 2026-07-17 从本轮比赛
> 主门撤出；没有照片不阻塞库存、空间、布局和任务卡技术闭环。
> 拍摄人:队友(新家现场)。填表与发射:任何人,照片到手后一键。

照片只回答“任务卡是否真的在物理世界执行”。当前确定性执行器不会直接解析图片
像素，而是消费与 `photo_ref` 绑定的人工 `present` 和 `region_id` 事实，再检查物品
出现与区域声明是否同时成立。因此照片与如实填表缺一不可；未执行本步骤时不得声称
真实房间已经复原或 `PHYSICAL_EXECUTION_VERIFIED`。

## 1. 摆放与拍摄

只验成果页当前展示的代表任务
(`results/hero/s1-auto-final-v1/taskcards/taskcards.md`):

- **card-02 学习文具箱**:书、水彩笔、铅笔与圆珠笔、转笔刀、剪刀、三角尺、
  壁纸刀 → 书桌(`auto_study_desk_01`)。

拍摄要求:

1. 七类物品按卡摆上书桌后,正面拍一张,**整个书桌区域入画**、光线充足、无遮挡;
2. 文件放 `local-data/hero_s1/acceptance/study_desk_after.jpg`(`local-data` 已
   gitignore,照片不入库、不发送云 API,只定向同步到赛方 Spark 做本次验收);
3. 如需分卡拍摄或补拍备选区域,复制 photos 数组条目即可(photo_ref 不得重复)。

## 2. 填表(如实,不粉饰)

1. 复制 `fixtures/hero_s1/acceptance.template.json` → `fixtures/hero_s1/acceptance.json`;
2. 保持 `selected_card_ids=["card-02"]`,未选四张卡不会进入本次验收,也不会被
   改写为 `FAILED`;
3. 逐实体把 `present` 改为照片中的真实情况(模板预填 `false`,失败安全:不改表跑出来
   的是 FAILED 而不是假 VERIFIED);
4. `match_source` 保持 `manual`(人工勾选);`match_score` 可选,人工确信可不填;
5. 低置信/缺件的结局是 NEEDS_USER / FAILED——这是设计功能不是事故;裁决走
   `adjudications`(`accept_override` / `reject_redo` + note),不改照片事实。

## 3. 发射

```bash
# 先部署已提交代码；随后把 verify.enabled 与 trace.strict 翻为 true。
scripts/deploy.sh
.venv/bin/python scripts/hero_pipeline.py \
  --config configs/hero_pipeline_s1_final.yaml \
  --run-dir results/hero/s1-auto-final-v1 \
  --from-stage verify --poll-interval 1
```

预期:healthcheck 先通过,随后验收清单、代表卡、父 trace 与照片被小体积定向同步到
Spark;EXEC/MEM/SPACE 在节点内运行,角色 fragments、verdict 与 fan-out telemetry
逐文件拉回。verify 产出三结局之一(VERIFIED / FAILED / NEEDS_USER),trace 严格模式
要求验收消息族闭合;成果页 `results/hero/s1-auto-final-v1/index.html` 验收复核区块更新。
该结果只扩展物理执行证据,不改变既有技术闭环结论。
