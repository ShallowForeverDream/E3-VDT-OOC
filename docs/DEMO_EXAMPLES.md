# 展示系统演示样例

当前 Gradio demo 使用轻量可解释规则管线，图片可先不传，重点填写：

- `新闻文本 / Caption`：被检测的新闻标题、caption 或 claim。
- `图像上下文`：图像原始 caption、OCR、检索证据或人工描述。

> 下面样例已加入 `demo/app.py` 的 Examples，启动系统后可直接点击。

| 场景 | 新闻文本 / Caption | 图像上下文 / Evidence | 预期输出重点 |
|---|---|---|---|
| 正常匹配 | `A flood caused evacuations in Shanghai in 2024.` | `A flood caused evacuations in Shanghai in 2024.` | `Non-OOC`，无强冲突字段 |
| 地点错配 | `A large protest erupted in Paris on Monday after a new climate policy.` | `People gathered in London during a climate demonstration on Monday.` | `OOC`，`location mismatch`，冲突字段 `location` |
| 时间错配 | `A protest took place in Paris in 2024.` | `People gathered for a protest in Paris on Monday.` | `OOC`，`temporal mismatch`，冲突字段 `time` |
| 人物/实体错配 | `Barack Obama will meet officials in Beijing in 2024.` | `Elon Musk will meet officials in Beijing in 2024.` | `OOC`，`entity mismatch`，冲突字段 `entity` |
| 事件类型错配 | `A covid hospital opened in Paris in 2024.` | `A football match took place in Paris in 2024.` | `OOC`，`event-type mismatch`，冲突字段 `event_type` |
| 行为关系错配 | `Soldiers attacked a convoy in Ukraine in 2024.` | `Soldiers rescued a convoy in Ukraine in 2024.` | `OOC`，`relation mismatch`，冲突字段 `relation` |
| 证据不足 | `A fire broke out in New York in 2024.` | 留空 | `Uncertain`，提示图像侧上下文不足 |

## 答辩演示顺序建议

1. 先点“正常匹配”，说明系统并不是一律报错。
2. 再依次展示地点、时间、实体、事件类型、行为关系五类错配。
3. 最后展示“证据不足”，强调系统会在图像上下文缺失时输出 `Uncertain`，避免伪造解释。
4. 讲解右侧 JSON：`mismatch_type`、`conflict_fields`、`event_scores` 正是本项目区别于普通二分类复现的结构化输出。
