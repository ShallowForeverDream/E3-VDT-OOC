# OOC 可解释检测相关工作梳理

本页用于统一项目口径：我们不再把当前规则解释模块包装成“已验证的新模型”，而是把它定位为 **VDT baseline 之上的 context-grounded attribution extension**，并通过后续实验验证其有效性。

## 1. VDT：跨域 OOC 二分类 baseline

VDT 的核心目标是解决 NewsCLIPpings 中不同新闻机构域之间的跨域 OOC 检测问题。其贡献集中在 domain-invariant variational representation、domain consistency constraint 和 test-time training。它适合作为本项目主分类 baseline，但它本身不输出 `entity/location/time/event_type/relation` 等错配归因字段。

**本项目使用方式**：

- VDT 负责 `OOC / Non-OOC` 主分类；
- 我们不声称重新提出优于 VDT 的主分类模型；
- 我们的新增部分聚焦 VDT 没有覆盖的解释和归因。

## 2. SNIFFER：MLLM + external evidence 的解释型 OOC 检测

SNIFFER 面向 explainable OOC misinformation detection，使用多模态大模型、instruction tuning 和外部证据进行 contextual verification。它说明 OOC 解释不应只靠图文相似度，而应结合图像真实上下文或外部证据。

**本项目借鉴点**：解释需要证据支撑。  
**本项目不做的事**：不复现 SNIFFER 的完整大模型指令微调流程。

## 3. COVE：先获得图像真实上下文，再判断 caption

COVE 的关键思路是 context-first：先预测或恢复图像真实上下文，再判断当前 caption 是否被该上下文支持。这个思路非常适合我们的系统，因为 VisualNews / NewsCLIPpings 本身包含图像原始新闻上下文。

**本项目落地为 COVE-lite**：

```text
NewsCLIPpings current caption
+ VisualNews original caption/title/context
-> event-field comparison
-> mismatch_type + conflict_fields
```

即：不依赖 MLLM 生成解释，而是用图像真实上下文作为证据，比较当前 caption 与真实上下文的事件字段差异。

## 4. MUSE：similarity shortcut 警告

MUSE 指出，简单相似度特征也可能在 OOC benchmark 上表现很强，因此不能仅凭总体 Accuracy/F1 证明模型理解事实一致性。项目中的 hard negative 评测就是为了回应这一点。

**本项目实验要求**：

- 构造 high-similarity OOC / same-topic different-event 样本；
- 报告 attribution 指标，而不只报分类指标；
- 比较 majority / random / text-only / similarity-only / COVE-lite rule。

## 5. AMG 与多粒度归因：细粒度标签需要标注与评测

AMG 类工作提醒我们：很多多模态假新闻数据集只有二分类标签，缺少 attribution labels。若要声称错配类型可靠，必须构造标注协议、人工校验集和量化指标。

**本项目必须补的实验**：

```text
examples/attribution_eval_set.jsonl
outputs/attribution_eval_metrics.json
```

指标至少包括：

- mismatch type accuracy；
- conflict field micro-F1；
- conflict field macro-F1；
- exact match rate。

## 6. 当前项目最终定位

准确表述：

> 本项目完成 VDT strict BLIP-2/GaussianBlur 核心复现，并在 VDT 二分类 baseline 上实现一个 COVE-lite true-context attribution extension。该扩展不覆盖 VDT 主分类结果，而是利用图像真实上下文与当前 caption 的事件字段差异输出错配类型、冲突字段和解释，并通过人工归因集与 hard negative 子集评估解释可靠性。

不要说：

- “E3-VDT 分类性能显著超过 VDT”；
- “我们完整实现了 Evidence Gate 和 Event-Guided TTT”；
- “弱监督标签天然正确”；
- “解释已经等价于 SNIFFER/COVE 级别”。

可以说：

- “我们完成了 VDT baseline 复现”；
- “我们将 VDT 从二分类输出扩展到 context-grounded attribution 输出”；
- “解释模块是否有效由人工归因评测集和 field-F1 支撑”。
