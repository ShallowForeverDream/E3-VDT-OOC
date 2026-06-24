# 创新点固定稿

## 总体创新

在 VDT 跨域 OOC 检测框架基础上，引入事件字段一致性和弱监督错配归因，使系统不仅判断图文是否错配，还能输出“错在哪里”。

## 创新点 1：事件字段一致性建模

VDT 主要学习 domain-invariant feature，但没有显式判断图文是否描述同一事件。我们引入五类事件字段：entity、location、time、event-type、relation，形成事件一致性向量：

```text
C_event = [s_entity, s_location, s_time, s_event_type, s_relation]
```

## 创新点 2：弱监督错配类型构造

现有 OOC 数据集通常没有细粒度错配类型标签。我们基于 NewsCLIPpings 构造方式、VisualNews 原始上下文、NER/时间地点抽取、规则与人工抽样校验构造弱监督标签。

标签集合：entity mismatch、location mismatch、temporal mismatch、event-type mismatch、relation mismatch、context omission、uncertain / evidence insufficient。

## 创新点 3：结构化错配归因输出

输出从二分类扩展为 label + mismatch_type + conflict_fields + event_scores + explanation。相比纯自然语言解释，结构化输出更稳定、更容易评价、更适合答辩展示。

## 创新点 4：Hard Negative 评测协议

构建 same-topic different-event、same-person different-time、same-location different-event、same-event-type different-location 等高相似错配样本，回应 MUSE 提出的 similarity shortcut 问题。


## 重要约束：分类准确率不降

本项目的创新不以牺牲 OOC / Non-OOC 分类准确率为代价。默认采用 **baseline-preserving sidecar**：VDT baseline 负责主分类，事件字段一致性模块负责输出 mismatch type、conflict fields、event scores 和 explanation。这样分类指标可以与 VDT 持平，同时显著增强可解释性。

如果后续尝试 event score 与 VDT score 融合，只有在验证集 Accuracy/F1 不低于 VDT baseline 时才作为主结果；否则只作为消融实验记录。

## 可选创新点：Event-guided TTT

如果时间允许，在 VDT 的 confidence + variance 伪标签筛选基础上加入 event stability。

## 不要写成创新点的内容

下载数据、复现 VDT、使用 BLIP-2、做 Gradio demo、报告 Accuracy/F1 都是工程基础，不是核心创新。
