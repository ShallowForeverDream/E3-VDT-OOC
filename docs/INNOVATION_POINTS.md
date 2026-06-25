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

## 创新点 3：Accuracy-preserving sidecar

正式策略中：

```text
final_label = vdt_label
final_score = vdt_score
```

事件归因模块只输出解释，不修改主分类。因此分类 Accuracy/F1 与 VDT baseline 持平；新增的是可解释诊断能力。

## 创新点 4：人工归因评测协议

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
- attribution head 训练；
- 自动 OCR / captioning。

这些可以放在后续工作，不能写成当前已完成主贡献。

## 不要写成创新点的内容

- 下载数据；
- 复现 VDT；
- 使用 BLIP-2；
- 做 Gradio demo；
- 只报告 Accuracy/F1；
- 没有人工验证的规则标签。
