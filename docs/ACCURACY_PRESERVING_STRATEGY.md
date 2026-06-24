# Accuracy-Preserving Strategy

本项目的核心约束：**E3-VDT 的 OOC / Non-OOC 分类准确率不能低于 VDT baseline**。创新优先体现在结构化错配归因与可解释性，而不是牺牲分类性能换解释。

## 设计原则

### 1. Baseline-preserving 主路径

最终系统默认采用 VDT baseline 的二分类结果作为主预测：

```text
final_label = vdt_label
final_score = vdt_score
```

事件字段一致性模块不直接覆盖分类结果，而是旁路输出：

```text
mismatch_type
conflict_fields
event_scores
explanation
```

因此在该模式下，E3-VDT 与 VDT 的分类 Accuracy / F1 理论上保持一致，但额外提供错配类型和解释。

### 2. Guarded fusion 只在验证集提升时启用

如果后续尝试将事件字段分数融合进分类：

```text
final_score = alpha * vdt_score + (1 - alpha) * event_consistency_score
```

必须满足以下条件才可启用：

```text
Acc(E3-VDT-fusion) >= Acc(VDT-baseline)
F1(E3-VDT-fusion)  >= F1(VDT-baseline) - tolerance
```

否则回退到 baseline-preserving 模式。

### 3. 归因头不反向破坏分类头

若做训练版 attribution head，推荐：

- 冻结或半冻结 VDT backbone。
- 分类 loss 仍为主 loss。
- attribution / mismatch-type loss 使用较小权重。
- 使用 validation gate：如果分类指标下降，则不采用该 checkpoint。

示意：

```text
L_total = L_cls + lambda_attr * L_attr
lambda_attr <= 0.1
```

### 4. 评价指标分层

报告中将指标分成两层：

| 层级 | 指标 | 目标 |
|---|---|---|
| 分类层 | Accuracy / F1 / AUC | 不低于 VDT baseline |
| 归因层 | mismatch type accuracy / conflict-field F1 / explanation case study | 显著优于 baseline，因为 baseline 不输出这些结构化字段 |

## 与 baseline 的公平比较

| 能力 | VDT baseline | E3-VDT baseline-preserving |
|---|---|---|
| OOC 二分类 | 有 | 有，默认沿用 VDT |
| Accuracy / F1 | baseline | 持平 |
| mismatch type | 无 | 有 |
| conflict fields | 无 | 有 |
| event scores | 无 | 有 |
| explanation | 弱/无结构 | 结构化、字段级 |
| Hard Negative 诊断 | 弱 | 强 |

## 报告口径

推荐表述：

> 本项目不以牺牲分类准确率换取解释性。E3-VDT 采用 baseline-preserving 策略：主分类结果默认继承 VDT baseline，从而保证 Accuracy/F1 不下降；在此基础上增加事件字段一致性建模与错配归因输出，使系统从“是否错配”的二分类扩展到“哪里错、为什么错、属于哪类错配”的结构化内容安全分析。

## 实验验收标准

1. VDT baseline 复现：记录 Accuracy / F1 / AUC。
2. E3-VDT sidecar 模式：分类结果与 VDT baseline 完全一致，Accuracy/F1 持平。
3. E3-VDT fusion 模式：只有在验证集分类指标不低于 baseline 时才进入主结果表，否则作为失败消融或不采用。
4. 创新性主要通过归因输出、Hard Negative 分析和案例解释体现。
