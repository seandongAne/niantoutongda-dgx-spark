# 任务 A(开发)锚点清单 — 2026-07-15 冻结

> 数据主责:Sean。17 件(≥15 达标)。编号只存在于本清单,画面中不可读。
> 任务 A 为开发用,不封存;任务 B 的对应清单由数据主责封存,技术侧不可见。

| # | 锚点(中文) | 检测词表(英文 class) | 相似对 |
|---|---|---|---|
| 1 | 玩具柜 | cabinet | C 组(柜类) |
| 2 | 玩具行李箱 | suitcase | B 组 |
| 3 | 玩具冰箱 | refrigerator | |
| 4 | 黑色书架 | bookshelf | A 组 |
| 5 | 蓝色水壶 | water bottle | D 组 |
| 6 | 夜灯 | night light | |
| 7 | 摄像头 | camera | |
| 8 | 书桌 | desk | |
| 9 | 台灯 | lamp | |
| 10 | 玫红色水壶 | water bottle | D 组 |
| 11 | 白色书架 | bookshelf | A 组 |
| 12 | 史迪奇玩偶 | stuffed animal | |
| 13 | 白色立柜 | cabinet | C 组 |
| 14 | 收纳盒 | storage box | |
| 15 | 脏衣篓 | laundry basket | |
| 16 | 粉色行李箱 | suitcase | B 组 |
| 17 | 床 | bed | |

困难负样本对(同类不同实例,S3 的考题):
- A 组:黑色书架 vs 白色书架(bookshelf ×2)
- B 组:玩具行李箱 vs 粉色行李箱(suitcase ×2)
- C 组:玩具柜 vs 白色立柜(cabinet ×2)
- D 组:蓝色水壶 vs 玫红色水壶(water bottle ×2)

检测词表 v5(2026-07-15 定稿,补拍段全项验证;box_threshold 0.28,批次=4,**词序即分批,易混概念强制分批**):

```
desk,water bottle,bookshelf,security camera,bed,tumbler,storage box,baby monitor,laundry bag,smart speaker,toy storage organizer,table lamp,luggage,mini fridge,wardrobe,stuffed animal,cylinder lamp
```

演进记录:
- v4:夜灯检测词 = **smart speaker**(白色柱体+贴纸,关灯态形似智能音箱;"night light" 全程零命中);摄像头双词 security camera(近摄)+ baby monitor(中距);批次 6→4(行李箱 0.61 分仍被 6 词批吞没);
- v5:**luggage 与 laundry bag 必须分批**(v4 同批再度互吞,分离后恢复双长轨);夜灯加备用词 **cylinder lamp** 排在第 17 位单独成批——它会把所有圆柱物都网进来,词表管召回,身份精度交给 S3 嵌入匹配+真值;
- 原则沉淀:**GDINO 的词表是召回网,不是分类器**——同批词竞争 token 归因,易混概念同批 = 互吞或复合标签;每批 ≤4 词、五个"柜架箱"类每批至多一个。

词表诊断记录:
- "cabinet" 对格架+抽屉盒式玩具柜零检出 → "toy storage organizer"(v2 起采用);
- 白色立柜首段怼脸拍导致物体大于画面 → 重拍退后入画即成轨(拍摄距离,非词表);
- 措辞胜负局:suitcase→**luggage**、toy refrigerator→**mini fridge**、desk lamp→**table lamp**、
  laundry basket→**laundry bag**(软袋非硬篓);玫红色水壶=保温杯型 → **tumbler**;
- **GDINO 得分稀释陷阱**:15 类一次喂,小类得分被稀释到阈值下(水壶 15 类碎片/3 类 0.56 分)
  → detect.py 改分批检测(≤6 类/批)+ 跨批 IoU>0.8 去重,阈值回到 0.30;
- 复合标签碎片("wardrobe bookshelf" 等)为批内相邻词伪影,量小可容忍;
- **夜灯唯一未决**:疑似发光球形灯,开灯状态过曝成白团任何词零命中——拍摄时关闭夜灯(或开大房间主灯)再验证。
