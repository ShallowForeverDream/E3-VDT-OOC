# 创新点固定稿：最终答辩版

## 0. 总体定位

本项目不声称已经提出一个超过 VDT 的新主分类模型。最终定位是：

> **VDT 负责 OOC / Non-OOC 主分类；本项目在其上构建可训练、可评测、可演示的错配归因 sidecar。COVE-lite true-context attribution 作为 oracle / 构造与评测辅助；最终应用路线是 VDT-CF-Attr no-true-context image+caption attribution head。**

其中：

```text
训练阶段可以利用 true context / 原始匹配样本构造标签；
推理阶段不输入 true_image_context，只输入 image + current_caption。
```

---

## 1. 已完成基础

- 完成 VDT strict BLIP-2/GaussianBlur 两组核心复现；
- 完成 no-true-context image+caption attribution head；
- 完成 COVE-lite oracle / true-context 辅助归因链路；
- 完成 controlled counterfactual attribution data 构造；
- 完成 100 条真实 OOC 人工归因评测集导入与统计；
- 完成 Gradio 单页极简 demo；
- 完成 accuracy-preserving sidecar：解释模块不覆盖 VDT 主分类结果。

---

## 2. 创新点一：Controlled Counterfactual Attribution Data

原始 NewsCLIPpings 只有 OOC / Non-OOC 二分类标签，不提供错配原因标签。为解决“归因模块没有监督标签”的问题，本项目从 Non-OOC 样本出发构造可控反事实数据：

```text
Non-OOC image-caption pair
  -> 只替换 current_caption 中一个实体 / 地点 / 年份 span
  -> 得到 entity / location / temporal mismatch gold label
```

当前已实现：

```text
entity_swap   -> entity mismatch
location_swap -> location mismatch
time_swap     -> temporal mismatch
none          -> benign illustrative image
```

这部分贡献是：**在无细粒度归因标签的数据集上，构造可控、可训练、可检查泄漏的错配原因监督信号。**

---

## 3. 创新点二：基于真实 OOC 人工分析的 different-event 分布修正

项目进一步构建了 100 条真实 OOC 人工归因评测集。统计显示：

```text
different-event mismatch 占 85%
真实 OOC 多为 entity / event_type / relation / location / time 多字段复合冲突
```

这说明真实 OOC 并不主要是单字段 location/time/entity 错配。基于这一发现，项目在训练集中加入严格筛选的原始 OOC low-similarity 样本作为 `different-event mismatch` 训练数据。

筛选原则：

```text
1. 原始样本必须是 OOC；
2. NewsCLIPpings similarity_score <= 0.65；
3. current_caption 与 true_image_context token Jaccard <= 0.08；
4. caption / true_context 至少 4 个 token；
5. 排除人工 gold set 中的 sample_id / text_id / image_id，避免训练评估泄漏。
```

这部分贡献是：**用真实 OOC 人工统计反向修正反事实训练分布，而不是停留在拍脑袋构造单字段样本。**

---

## 4. 创新点三：VDT-CF-Attr no-true-context attribution head

最终应用路线不依赖 true context。

```text
推理输入：image + current_caption
系统内部：VDTAdapter -> OOC / Non-OOC / Uncertain
归因输出：mismatch_type + conflict_fields + confidence
```

Attribution head 使用的特征包括：

```text
CLIP image-caption similarity
CLIP image-field-prompt similarity
caption field presence/count
caption length/token count
VDT score
```

字段 prompt 示例：

```text
entity   -> a news photo involving Biden
location -> a news photo taken in Paris
time     -> a news photo from 2024
event    -> a news photo about protest
relation -> a news photo showing speak
```

该模块的意义是：**数据集外推理不要求提供 VisualNews 原始 caption 或 true context，更接近真实应用场景。**

---

## 5. 创新点四：Oracle / no-true-context 双评测体系

项目明确区分两种设置：

| 设置 | 推理时是否使用 true context | 作用 |
|---|---:|---|
| COVE-lite true-context oracle | Yes | 上限分析、构造辅助、人工评测参考 |
| VDT-CF-Attr no-true-context head | No | 最终应用路线 |

这种区分避免把 oracle 结果误写成真实应用结果。

---

## 6. 创新点五：真实 OOC 100 条人工归因评测集

为了回答“解释是否真的正确”，项目构建了两批共 100 条真实 OOC 人工归因评测集。

标注内容：

```text
gold_mismatch_type
gold_conflict_fields
rationale / rationale_warning
annotation_status
```

评测指标：

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
per-type confusion
per-field F1
```

该评测集用于验证 no-true-context 模型是否能从合成反事实训练泛化到真实 OOC 场景。

---

## 7. 当前关键实验结果

### 7.1 VDT baseline

| 实验 | F1 | Acc | AUC |
|---|---:|---:|---:|
| bbc,guardian, bs128 | 0.7353 | 0.7383 | 0.7398 |
| usa_today,washington_post, bs64 | 0.8032 | 0.8032 | 0.8028 |

### 7.2 no-true-context scaling

| MaxPerType | Best stable method | Type Acc | Field Micro-F1 | Exact Match |
|---:|---|---:|---:|---:|
| 80 | logistic regression no-true-context | 0.2745 | 0.3564 | 0.1961 |
| 200 | logistic regression no-true-context | 0.4266 | 0.5195 | 0.2308 |
| 1000 | logistic regression no-true-context | 0.5275 | 0.5719 | 0.3250 |

结论：随着可控反事实训练数据增加，no-true-context attribution head 的性能提升。

### 7.3 plus2000 different-event setting

训练分布：

```text
none/entity/location/time/different-event = 1000/1000/1000/1000/3000
```

合成 held-out test：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.4669 | 0.7012 | 0.4669 |
| logistic regression no-true-context | 0.3317 | 0.6655 | 0.4228 |
| image+caption MLP attribution head | **0.5220** | 0.6876 | 0.3487 |

### 7.4 真实 OOC 100 条 no-true-context 评估

| Model | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| 5way_1000 | 0.0900 | 0.3276 | 0.0300 |
| 5way_plus2000 | **0.2900** | **0.4781** | 0.0300 |

结论：加入额外 original-OOC different-event 训练样本后，真实 OOC 评估明显改善，但仍不能声称真实 OOC 归因已经完全可靠。

---

## 8. 创新点六：字段存在性约束后处理

no-true-context 推理容易出现一个问题：caption 中没有某字段，但模型输出该字段错配。例如 caption 中没有 entity，却输出 entity mismatch。

项目已加入 field-presence constraint：

```text
如果 entity_present=0，则不能输出 entity mismatch；
如果 location/time/event_type/relation 不存在，同理不能输出对应 mismatch；
无有效字段时，OOC 输出 uncertain / insufficient visual evidence，Non-OOC 输出 benign illustrative image。
```

输出 JSON 包含：

```json
{
  "postprocess_applied": true,
  "postprocess_reason": "field_absent_constraint_reselected_from_present_fields"
}
```

这使前端 demo 更符合逻辑，也更容易答辩。

---

## 9. 创新点七：极简单页交互系统

当前前端已经改为单页应用：

```text
输入：图片上传 + 文字输入
输出：一张结果卡片
```

结果卡片集中展示：

```text
OOC 判定
ooc 总可信度
错配类型
错配类型分数
冲突字段表
```

前端默认不展示 true context 输入，突出最终应用路线。

---

## 10. 不要夸大的内容

不要写：

```text
1. 我们超过 VDT 主分类性能；
2. 我们已经彻底解决真实 OOC 错配归因；
3. COVE-lite oracle 结果就是最终应用结果；
4. 原始 OOC 全部天然是 different-event gold；
5. Gradio demo 本身是算法创新。
```

应该写：

```text
1. VDT 主分类已复现，解释模块作为 sidecar；
2. controlled counterfactual data 提供可训练错配原因标签；
3. no-true-context head 是最终应用路线；
4. COVE-lite 是 oracle / 构造与评测辅助；
5. 真实 OOC 评估显示方向有效，但泛化仍有挑战。
```

---

## 11. 最稳答辩口径

> 我们完成了 VDT strict baseline 的核心复现。VDT 负责 OOC / Non-OOC 主分类。由于原始 NewsCLIPpings 没有错配原因标签，我们首先从 Non-OOC 样本构造可控反事实数据，通过单字段替换得到 entity/location/time 的归因标签。随后通过 100 条真实 OOC 人工标注发现真实错配以 different-event 为主，因此进一步加入严格筛选的低相似原始 OOC 样本作为 different-event 训练数据。最终应用路线是 VDT-CF-Attr no-true-context head，推理阶段只输入 image + current_caption，不使用 true_image_context。实验显示，加入 different-event 训练后真实 OOC 100 条评估从 Type Acc 0.09 提升到 0.29，Field Micro-F1 从 0.328 提升到 0.478，证明训练分布修正有效；但真实 OOC 多字段复合错配仍较难，后续需要更强视觉语义、OCR、检索或 VLM caption 增强。
