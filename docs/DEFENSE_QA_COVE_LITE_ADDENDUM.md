# 答辩补充口径：COVE-lite 归因路线

## Q1：你们是不是只是用了 VDT，然后加了一些解释标签？

推荐回答：

> 是的，主分类能力来自 VDT。我们没有声称重新提出超过 VDT 的主分类模型。我们的工作是在 VDT 之上做 COVE-lite true-context attribution：利用 VisualNews 原始上下文作为图像真实语境，比较它和当前 caption 的事件字段差异，输出错配类型和冲突字段。

## Q2：怎么知道你们输出的错配类型是对的？

推荐回答：

> 不能只靠规则说它对。因此我们新增人工归因评测集，标注 `gold_mismatch_type` 和 `gold_conflict_fields`，并比较 majority、sampled、text-only 和 COVE-lite event rule。只有 COVE-lite 在 field-F1 上超过简单 baseline，我们才把它作为有效解释贡献。

## Q3：你们的创新点到底是什么？

推荐回答：

> 创新点不是主分类涨分，而是把 VDT 从二分类输出扩展为基于真实上下文的错配归因系统。VDT 判断是否 OOC；COVE-lite attribution 判断错在哪里，并给出结构化字段。

## Q4：Evidence Gate 和 Event-Guided TTT 做了吗？

推荐回答：

> 这两个是扩展方向。当前已完成的是 VDT baseline 复现、COVE-lite 真实上下文归因脚本和可解释 demo。Evidence Gate 与 Event-Guided TTT 不作为当前已完成主贡献。

## Q5：为什么需要 VisualNews metadata？

推荐回答：

> 因为要避免解释凭空生成。VisualNews metadata 提供图片原始新闻语境，我们用它作为 true image context，再与当前 caption 比较，这比手写 image_context 更可验证。

## Q6：最终效果怎么报告？

推荐回答：

分类层面：报告 VDT 两组复现指标。  
归因层面：报告人工归因集上的 type accuracy、field micro-F1、macro-F1 和 exact match。

如果归因指标没有明显超过 baseline，必须如实写失败分析，而不是强行说有效。
