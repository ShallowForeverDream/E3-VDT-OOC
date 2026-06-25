# 创新点固定稿：实事求是版

## 总体定位

本项目不再声称已经提出一个超过 VDT 的新主分类模型。当前更准确的定位是：

> **VDT 负责 OOC / Non-OOC 主分类；E3-VDT-OOC 在 VDT 之上提供 COVE-lite true-context attribution，用于输出错配类型、冲突字段、事件分数和结构化解释。**

## 已完成基础

- 完成 VDT strict BLIP-2/GaussianBlur 两组核心复现；
- 完成 Gradio demo 与统一 JSON schema；
- 完成 accuracy-preserving sidecar：解释模块不覆盖 VDT 主分类结果。

## 创新点 1：COVE-lite true-context attribution

现有 VDT 只能输出 `OOC / Non-OOC`，不能说明错在哪里。我们借鉴 COVE 的 context-first 思路，用 VisualNews 原始上下文作为图像真实语境：

```text
current caption vs true image context
```

再比较五类事件字段：

```text
entity / location / time / event_type / relation
```

输出：

```text
mismatch_type + conflict_fields + event_scores + explanation
```

## 创新点 2：面向无细粒度标签场景的弱归因标签构造

NewsCLIPpings 主要提供二分类标签，缺少 location mismatch、temporal mismatch 等细粒度标签。我们用事件字段冲突生成弱归因标签，但该标签不直接当作真值，必须通过人工评测集验证。

## 创新点 3：Controlled Counterfactual Attribution

为了避免“从原始 OOC 样本里硬猜错配原因”的脏标签问题，我们新增了一个可控反事实构造流程：

```text
Non-OOC 样本
  -> 只替换一个 caption span
  -> 得到 entity/location/time 单字段错配
  -> 自动获得 gold_mismatch_type 和 gold_conflict_fields
```

当前已实现：

- `entity_swap -> entity mismatch`
- `location_swap -> location mismatch`
- `time_swap -> temporal mismatch`
- `none -> benign illustrative image`

这部分贡献的准确表述是：**构造可控单字段错配的 attribution 训练/测试集**，不是声称原始 OOC 数据集已经自带细粒度原因标签。

## 创新点 4：轻量 Attribution Head（oracle 特征版）

在可控反事实数据上，我们训练一个轻量 MLP attribution head，输入不是原图，而是解释侧特征：

```text
field-wise NLI scores
+ evidence relevance
+ graph alignment features
```

当前一次 `MaxPerType=80` 本地实验中，修复泄漏后的 group split 在可控反事实 test=38 上：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.2368 | 0.2812 | 0.2368 |
| field-wise NLI | 0.7632 | 0.7200 | 0.6579 |
| attribution head MLP | 0.9474 | 0.9020 | 0.9211 |

这次结果已经过泄漏检查：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross-split duplicate edited caption = 0
```

注意：这只证明在“可控单字段反事实测试集”上有效。真实 OOC 泛化仍需要人工标注集评测，不能夸大成所有真实样本都已经解决。

## 创新点 5：VDT-CF-Attr no-true-context 推理路线

进一步修正后的最终应用路线是：

```text
训练阶段：
Non-OOC image-caption pair
  -> 单字段反事实编辑 caption
  -> 得到 mismatch_type gold label

推理阶段：
image + current_caption + VDT score
  -> image-caption attribution head
  -> mismatch_type / conflict_fields
```

关键边界：

```text
true_image_context 只能用于构造训练标签和人工标注参考；
不能作为最终推理阶段 attribution head 的必要输入。
```

当前已实现 `scripts/features/build_image_caption_attribution_features.py` 和 `scripts/train/train_no_true_context_attribution_head.py`。本地 `MaxPerType=80`、CLIP image+caption 特征、group split、test=38 的 no-true-context 结果：

| Method | Uses true context at inference? | Type Acc | Field Micro-F1 | Exact Match |
|---|---|---:|---:|---:|
| field prompt grounding rule | False | 0.3421 | 0.3415 | 0.3947 |
| logistic regression no-true-context | False | 0.4474 | 0.3939 | 0.2632 |
| image+caption MLP attribution head | False | 0.4474 | 0.3500 | 0.4474 |

这组结果低于 COVE-lite oracle attribution，是合理的：没有 true context 时，具体人物、地点、年份很多时候无法仅靠图像可靠判断。它反而更适合作为真实应用路线的诚实基线。

## 创新点 6：Accuracy-preserving sidecar

正式策略中：

```text
final_label = vdt_label
final_score = vdt_score
```

事件归因模块只输出解释，不修改主分类。因此分类 Accuracy/F1 与 VDT baseline 持平；新增的是可解释诊断能力。

## 创新点 7：人工归因评测协议

为了回答“你怎么知道解释是对的”，项目新增人工归因评测集和 baseline 对比：

```text
majority / sampled / text-only / COVE-lite event rule
```

指标：

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

只有当 COVE-lite event rule 在人工集上超过简单 baseline，才把归因模块写成有效贡献。

## 可选扩展，不作为当前已完成贡献

- Evidence Gate；
- Event-Guided TTT；
- 自动 OCR / captioning。

这些可以放在后续工作，不能写成当前已完成主贡献。

## 不要写成创新点的内容

- 下载数据；
- 复现 VDT；
- 使用 BLIP-2；
- 做 Gradio demo；
- 只报告 Accuracy/F1；
- 没有人工验证的规则标签。
