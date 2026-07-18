# 验收照片指引 · hero-s1 终判分跑(G7b)

> 目标:至少 1 张任务卡走完 presence ∧ compliance → VERIFIED 全消息链(预注册门 G7b)。
> 拍摄人:队友(新家现场)。填表与发射:任何人,照片到手后一键。

## 1. 摆放与拍摄

两张任务卡目标区域相同,一张照片即可覆盖(`results/hero/s1-final/taskcards/taskcards.md`):

- **card-01 组合1箱**:壁纸刀 → 红木斗柜台面(`chest_top`)
- **card-02 杯子箱**:白色马克杯 + 咖啡罐(玻璃罐装豆)+ 粉色保温水杯 → 红木斗柜台面(`chest_top`)

拍摄要求:

1. 四件物品按卡摆上红木斗柜台面后,正面拍一张,**整个台面入画**、光线充足、无遮挡;
2. 文件放 `local-data/hero_s1/acceptance/chest_top_after.jpg`(local-data 已 gitignore,照片不入库、不出境);
3. 如需分卡拍摄或补拍备选区域,复制 photos 数组条目即可(photo_ref 不得重复)。

## 2. 填表(如实,不粉饰)

1. 复制 `fixtures/hero_s1/acceptance.template.json` → `fixtures/hero_s1/acceptance.json`;
2. 逐实体把 `present` 改为照片中的真实情况(模板预填 `false`,失败安全:不改表跑出来的是 FAILED 而不是假 VERIFIED);
3. `match_source` 保持 `manual`(人工勾选);`match_score` 可选,人工确信可不填;
4. 低置信/缺件的结局是 NEEDS_USER / FAILED——这是设计功能不是事故;裁决走 `adjudications`(`accept_override` / `reject_redo` + note),不改照片事实。

## 3. 发射

```bash
# configs/hero_pipeline_s1_final.yaml:verify.enabled 翻 true,trace.strict 翻 true
.venv/bin/python scripts/hero_pipeline.py --config configs/hero_pipeline_s1_final.yaml --from-stage verify
```

预期:verify 产出三结局之一(VERIFIED / FAILED / NEEDS_USER),trace 严格模式要求验收消息族闭合;成果页 `results/hero/s1-final/index.html` 验收复核区块更新;G7a(一键复跑 hash 入 bundle)与 G7b 同跑取证。
