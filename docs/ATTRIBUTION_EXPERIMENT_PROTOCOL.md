# COVE-lite 错配归因实验协议

本协议定义后续必须补齐的解释性实验。目标不是证明我们超过 VDT 的主分类，而是验证：在 VDT 主分类结果之外，COVE-lite 真实上下文归因是否能提供可用的 `mismatch_type` 与 `conflict_fields`。

## 1. 核心定位

当前仓库已经完成：

- VDT strict BLIP-2/GaussianBlur 两组核心复现；
- Gradio demo；
- event sidecar heuristic；
- accuracy-preserving 输出策略。

但还不能声称：

- E3-VDT 分类性能超过 VDT；
- 规则错配标签天然正确；
- Evidence Gate / Event-Guided TTT 已完成。

后续实验只验证一个更实在的问题：

> 使用 VisualNews 原始上下文作为图像真实语境，比较 current caption 与 true image context 的事件字段差异，是否能比随机/多数类/text-only baseline 更准确地指出错配字段？

## 2. COVE-lite 数据构造

生成文件：

```text
outputs/cove_lite_context_pairs.jsonl
```

每行格式：

```json
{
  "sample_id": "...",
  "image_id": "...",
  "text_id": "...",
  "split": "test",
  "generator": "merged_balanced",
  "domain": "bbc",
  "label": 1,
  "current_caption": "当前 NewsCLIPpings caption / text context",
  "true_image_context": "VisualNews 中该 image_id 的原始 caption/title/context",
  "source": "visualnews_metadata"
}
```

解释：

- `current_caption`：当前图文对中的文本语境；
- `true_image_context`：图像在 VisualNews 中的原始语境；
- 对 OOC 样本，二者可能描述不同事件；
- 对 Non-OOC 样本，二者应更一致。

## 3. 弱监督归因标签

生成文件：

```text
outputs/weak_attribution_labels.jsonl
```

每行格式：

```json
{
  "sample_id": "...",
  "label": 1,
  "weak_mismatch_type": "location mismatch",
  "weak_conflict_fields": ["location"],
  "event_scores": {
    "entity": 0.8,
    "location": 0.1,
    "time": 0.5,
    "event_type": 0.7,
    "relation": 0.6
  }
}
```

该文件不是 gold label，只是自动规则输出，必须经过人工子集评估。

## 4. 人工评测集

生成候选文件：

```text
examples/attribution_eval_candidates.jsonl
```

人工标注后保存为：

```text
examples/attribution_eval_set.jsonl
```

每行格式：

```json
{
  "sample_id": "...",
  "current_caption": "...",
  "true_image_context": "...",
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "annotator": "A",
  "rationale": "current caption says Paris, true context says London"
}
```

推荐规模：先 100 条。最小可接受规模：50 条。

推荐构成：

| 类型 | 数量 |
|---|---:|
| Non-OOC | 20 |
| OOC high-similarity / hard negative | 30 |
| OOC normal | 30 |
| evidence insufficient / uncertain | 20 |

如果时间不够，先人工标 50 条，并在报告里说明限制。

## 5. Baseline 对比

必须比较：

| 方法 | 说明 |
|---|---|
| majority | 永远预测人工评测集中最常见 mismatch type |
| random | 随机预测 mismatch type / conflict field |
| text-only | 不使用 true_image_context，只根据 current_caption 输出 uncertain 或文本内部字段 |
| current heuristic | 当前 demo 规则，如果输入 image_context |
| COVE-lite event rule | current_caption vs true_image_context 的事件字段比较 |

可选：

| 方法 | 说明 |
|---|---|
| attribution head | 用弱标签训练 LogisticRegression / MLP；若时间不足不作为主结果 |

## 6. 指标

报告以下指标：

```text
mismatch_type_accuracy
conflict_field_micro_precision
conflict_field_micro_recall
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

定义：

- `mismatch_type_accuracy`：主错配类型是否一致；
- `conflict_field_micro_f1`：多标签字段级 F1；
- `exact_match_rate`：预测字段集合与 gold 字段集合完全一致的比例。

## 7. 成功标准

项目只在以下条件下声称 attribution 有效：

```text
COVE-lite event rule 的 field micro-F1 > majority / random / text-only baseline
```

如果没有超过，则必须如实写成失败分析。

## 8. 一键运行顺序

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC

# 1. 构造 COVE-lite 上下文对
python scripts/context/build_cove_lite_context_pairs.py `
  --newsclippings-data-dir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  --visualnews-metadata-dir E:\OOC_Datasets\VisualNews\metadata `
  --output outputs\cove_lite_context_pairs.jsonl

# 2. 构造弱归因标签
python scripts/labels/build_weak_attribution_from_context.py `
  --input outputs\cove_lite_context_pairs.jsonl `
  --output outputs\weak_attribution_labels.jsonl

# 3. 抽人工标注候选集
python scripts/eval/build_attribution_eval_sample.py `
  --context-pairs outputs\cove_lite_context_pairs.jsonl `
  --weak-labels outputs\weak_attribution_labels.jsonl `
  --output examples\attribution_eval_candidates.jsonl `
  --n 120

# 4. 人工标注 examples\attribution_eval_candidates.jsonl 为 examples\attribution_eval_set.jsonl

# 5. 评测归因 baseline
python scripts/eval/run_attribution_baselines.py `
  --gold examples\attribution_eval_set.jsonl `
  --weak-labels outputs\weak_attribution_labels.jsonl `
  --output outputs\attribution_eval_metrics.json
```

## 9. 报告写法

可以说：

> 我们实现了 COVE-lite true-context attribution，并在人工归因集上评估其 mismatch type accuracy 和 conflict field F1。

不能说：

> 我们证明了弱监督标签完全正确。
> 我们超过了 SNIFFER/COVE 的解释能力。
> 我们提出了新的主分类 SOTA 模型。
