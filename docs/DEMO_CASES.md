# Demo 测试与答辩演示样例

这些样例用于当前 Gradio demo 和命令行推理。当前 demo 不直接识图，`image_context` 代表图像原始 caption / OCR / 检索证据。

## 推荐演示顺序

1. 先跑 `ex01_non_ooc_same_event`，展示 Non-OOC 正常样例。
2. 再跑 `ex02` 到 `ex06`，分别展示 location/time/entity/event_type/relation 五类错配。
3. 跑 `ex07_multi_field_hard_negative`，展示 Hard Negative 和多字段归因。
4. 跑 `ex08_uncertain_no_image_context`，说明系统证据不足时会输出 Uncertain。
5. 如果老师想看中文，跑 `ex09_chinese_location_mismatch`。
6. 切到网页端 `分类不降验证` 标签页，展示 sidecar 可以发现冲突，但最终 label 仍继承 VDT baseline。

## 一键命令行测试

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
python scripts\run_demo_cases.py
```

## 样例表

| ID | 场景 | Text | Image context | 期望输出 | 演示要点 |
|---|---|---|---|---|---|
| `ex01_non_ooc_same_event` | 正例：图文描述同一事件 | A flood caused evacuations in Shanghai in 2024. | A flood caused evacuations in Shanghai in 2024. | Non-OOC / benign illustrative image / no conflict | 用于先展示系统正常输出：字段一致、整体判为 Non-OOC。 |
| `ex02_location_mismatch` | 地点错配：同类抗议事件但地点不同 | A large protest erupted in Paris on Monday after a new climate policy. | People gathered in London during a climate demonstration on Monday. | OOC / location mismatch / location | 适合讲 location 字段：事件类型和时间相近，但 Paris/London 冲突。 |
| `ex03_temporal_mismatch` | 时间错配：同一地点同类事件但时间不同 | A protest took place in Paris in 2024. | People gathered for a protest in Paris on Monday. | OOC / temporal mismatch / time | 适合讲 temporal 字段：地点和事件类型一致，但时间表达冲突。 |
| `ex04_entity_mismatch` | 主体错配：人物不同但地点时间动作一致 | Barack Obama will meet officials in Beijing in 2024. | Elon Musk will meet officials in Beijing in 2024. | OOC / entity mismatch / entity | 适合讲 entity 字段：Beijing/2024/meet 一致，但 Barack Obama 与 Elon Musk 冲突。 |
| `ex05_event_type_mismatch` | 事件类型错配：地点时间一致但新闻类型不同 | A covid hospital opened in Paris in 2024. | A football match took place in Paris in 2024. | OOC / event-type mismatch / event_type | 适合讲 event_type 字段：Paris/2024 一致，但 health 与 sports 冲突。 |
| `ex06_relation_mismatch` | 关系/动作错配：主体地点时间一致但动作不同 | Soldiers attacked a convoy in Ukraine in 2024. | Soldiers will meet a convoy in Ukraine in 2024. | OOC / relation mismatch / relation | 适合讲 relation 字段：主体、地点、时间、事件域一致，但 attacked/meet 动作冲突。 |
| `ex07_multi_field_hard_negative` | Hard Negative：跨多个字段冲突 | A fire broke out in New York on Monday. | A football match took place in London in 2019. | OOC / location mismatch / location,time,event_type | 适合压轴展示：地点、时间、事件类型同时冲突，系统仍输出主错配类型和冲突字段列表。 |
| `ex08_uncertain_no_image_context` | 证据不足：没有图像上下文 | A fire broke out in New York in 2024. | (empty) | Uncertain / uncertain / evidence insufficient / no conflict | 用于说明系统不会在证据不足时硬判，符合“证据约束”的创新点。 |
| `ex09_chinese_location_mismatch` | 中文地点错配：中文输入可演示 | 2024年，北京市举行抗议游行。 | 2024年，上海市举行抗议游行。 | OOC / location mismatch / location | 用于展示系统支持中文关键词和中文地点字段抽取。 |
| `ex10_non_ooc_finance_context` | 正例：财经场景一致 | stock market rose in New York in 2024. | stock market traders worked in New York in 2024. | Non-OOC / benign illustrative image / no conflict | 展示不是所有语义不完全相同都判 OOC；关键事件字段一致时可判 Non-OOC。 |

## 答辩口径

- 我们不是只输出一个真假标签，而是输出结构化字段：`mismatch_type`、`conflict_fields`、`event_scores`、`explanation`。
- 这些样例覆盖了从单字段错配到多字段 Hard Negative 的主要场景。
- 解释模块是 sidecar：正式实验中 `label = baseline_label`，因此分类 Accuracy/F1 不会被错配归因模块拉低。
- 当前样例用于系统展示；最终实验指标以 VDT/BLIP-2 严格复现与 E3-VDT 扩展实验为准。
