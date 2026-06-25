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

## 创新点 4：轻量 Attribution Head

在可控反事实数据上，我们训练一个轻量 MLP attribution head，输入不是原图，而是解释侧特征：

```text
field-wise NLI scores
+ evidence relevance
+ graph alignment features
```

当前一次 `MaxPerType=80` 本地实验中，在可控反事实 test=41 上：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.2927 | 0.0000 | 0.2927 |
| field-wise NLI | 0.6098 | 0.5714 | 0.6098 |
| attribution head MLP | 0.9268 | 0.9474 | 0.9512 |

注意：这只证明在“可控单字段反事实测试集”上有效。真实 OOC 泛化仍需要人工标注集评测，不能夸大成所有真实样本都已经解决。

## 创新点 5：Accuracy-preserving sidecar

正式策略中：

```text
final_label = vdt_label
final_score = vdt_score
```

事件归因模块只输出解释，不修改主分类。因此分类 Accuracy/F1 与 VDT baseline 持平；新增的是可解释诊断能力。

## 创新点 6：人工归因评测协议

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
