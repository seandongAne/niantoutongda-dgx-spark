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

检测词表 v2(2026-07-15 用第一段真实视频诊断后修订;一概念一词,颜色区分交给嵌入/属性层):

```
toy storage organizer,wardrobe,suitcase,toy refrigerator,bookshelf,water bottle,tumbler,night light,security camera,desk,desk lamp,stuffed animal,storage box,laundry basket,bed
```

词表诊断记录(视频 g0_a_old1,box_threshold 0.28):
- "cabinet" 对格架+抽屉盒式玩具柜零检出 → "toy storage organizer" 成 22 帧长轨(v2 采用);
- 白色立柜是整面白门,首段视频怼脸拍导致物体大于画面,任何词都框不出——拍摄距离问题非词表问题;
- 词组间避免共享单词(曾出现 "toy storage organizer storage box" 复合碎片),复合碎片被 min_track_len 过滤,可容忍;
- 玫红色水壶外观为保温杯型 → 词表补 "tumbler"。
