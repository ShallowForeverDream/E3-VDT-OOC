# 答辩 PPT 内容稿：VDT-COVE-Attr 版

> 本稿替代旧版“E3-VDT Full 分类提升”口径。PPT 中不要写 Evidence Gate / Event-Guided TTT 已完成，也不要写分类指标提升 X%。

## 1. 标题页

**题目**：VDT-COVE-Attr：跨域图文内容挪用检测与错配归因系统  
**副标题**：基于 VDT baseline 的 COVE-lite true-context attribution

讲稿：我们关注新闻场景中的图文内容挪用，即真实图片被放入错误文本语境中传播的问题。

## 2. 问题背景

- OOC 风险来自图文语境错配，不一定来自图像篡改。
- 图像和文本可能主题相似，但不是同一事件。
- 内容审核不仅要知道是否 OOC，还要知道错在哪里。

## 3. 任务定义

输入：新闻图文对。  
输出：

```text
OOC / Non-OOC
mismatch_type
conflict_fields
event_scores
explanation
```

## 4. 现有方法不足

| 方法/问题 | 局限 |
|---|---|
| VDT | 能做跨域 OOC 二分类，但不解释错配字段 |
| 相似度方法 | 可能只利用 surface shortcut |
| OOC 数据集 | 缺少细粒度错配类型标签 |
| 自由生成解释 | 如果没有证据 grounding，难以验证 |

## 5. 文献依据

| 工作 | 对本项目的启发 |
|---|---|
| VDT | 主分类 baseline，解决跨域 OOC 检测 |
| SNIFFER | 解释需要 contextual verification 和外部证据 |
| COVE | 先恢复图像真实上下文，再判断 caption |
| RED-DOT / CMIE | 证据相关性和证据噪声需要控制 |
| MUSE | 要防止 similarity shortcut |
| AMG | 归因标签需要人工评测 |

## 6. VDT baseline 复现

| 设置 | F1 | Acc | AUC | 状态 |
|---|---:|---:|---:|---|
| bbc,guardian bs128 | 0.7353 | 0.7383 | 0.7398 | completed |
| usa_today,washington_post bs128 | - | - | - | CUDA OOM |
| usa_today,washington_post bs64 | 0.8032 | 0.8032 | 0.8028 | completed |

口径：完成核心 strict setting 两组复现，不夸大为完整复现全部论文实验。

## 7. 方法总览：VDT-COVE-Attr

```text
VDT baseline -> OOC / Non-OOC
COVE-lite true context -> 图像真实语境
Event attribution sidecar -> mismatch_type / conflict_fields / explanation
```

关键约束：

```text
final_label = vdt_label
```

## 8. COVE-lite true context

```text
NewsCLIPpings sample
-> image_id
-> VisualNews original caption/title/context
-> true_image_context
```

然后比较：

```text
current_caption vs true_image_context
```

意义：解释基于图像原始语境，而不是系统自由编造。

## 9. 事件字段归因

比较五类字段：

```text
entity
location
time
event_type
relation
```

输出示例：

```json
{
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "event_scores": {"location": 0.1, "time": 0.8}
}
```

## 10. 为什么要人工评测

NewsCLIPpings 只有 OOC / Non-OOC 二分类标签，没有错配类型标签。  
因此自动归因标签不能直接当真，必须人工评估。

人工标注字段：

```text
gold_mismatch_type
gold_conflict_fields
rationale
```

## 11. 归因 baseline 对比

| 方法 | 说明 |
|---|---|
| majority | 永远预测最常见类型 |
| sampled | 随机采样类型 |
| text-only | 不看 true context |
| COVE-lite event rule | current caption vs true context 字段比较 |

指标：

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

## 12. Demo 展示

展示顺序：

1. Non-OOC；
2. location mismatch；
3. temporal mismatch；
4. event-type mismatch；
5. evidence insufficient；
6. accuracy-preserving 模式。

强调 JSON 输出中的 `decision_source=vdt_baseline`。

## 13. 当前完成情况

已完成：

- VDT 两组核心复现；
- Gradio demo；
- 结构化输出 schema；
- COVE-lite 数据构造脚本；
- 归因评测脚本。

本地待跑：

- `cove_lite_context_pairs.jsonl`；
- `weak_attribution_labels.jsonl`；
- 人工标注 50-100 条；
- `attribution_eval_metrics.json`。

## 14. 贡献总结

1. 完成 VDT strict 核心 baseline 复现；
2. 明确 VDT 不输出错配字段的不足；
3. 适配 COVE 的 context-first 思路，提出 VDT-COVE-Attr；
4. 设计人工归因评测协议，用 field-F1 验证解释；
5. 实现可展示的内容安全系统。

## 15. 不足与未来工作

- 事件抽取仍是轻量规则；
- 人工归因集规模有限；
- Evidence Gate / Event-Guided TTT / attribution head 训练作为后续工作；
- 当前不声称主分类超过 VDT。

## 16. 结尾话术

> VDT 判断是否 OOC；VDT-COVE-Attr 进一步解释哪里错。我们的贡献是把 VDT 从二分类扩展成基于真实上下文的可解释审核系统，并用人工归因评测检验解释是否可靠。
