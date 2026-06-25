# 实验计划与结果表模板

## 实验 0：输入检查

目的：确认 NewsCLIPpings JSON 与 VisualNews metadata 可读。

输出：

- `outputs/input_check.json`

## 实验 1：COVE-lite context coverage

目的：验证多少 NewsCLIPpings 样本能映射到 VisualNews true context。

指标：

| Metric | Value |
|---|---:|
| total samples |  |
| samples with current caption |  |
| samples with true context |  |
| coverage |  |

## 实验 2：事件抽取评估

人工标注 100 条 event tuple。比较：

| Extractor | Entity F1 | Location F1 | Time F1 | Event Type F1 | Relation F1 |
|---|---:|---:|---:|---:|---:|
| Rule baseline |  |  |  |  |  |
| NER/date/OpenIE |  |  |  |  |  |
| LLM JSON extractor |  |  |  |  |  |

## 实验 3：归因方法对比

人工标注 50–100 条 attribution gold set。比较：

| Method | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |
|---|---:|---:|---:|---:|
| Majority |  |  |  |  |
| Random / Sampled |  |  |  |  |
| Rule sidecar |  |  |  |  |
| Similarity-only |  |  |  |  |
| Field-wise NLI |  |  |  |  |
| Field-wise NLI + evidence relevance |  |  |  |  |

## 实验 4：evidence relevance ablation

| Variant | Type Acc | Field F1 | Evidence Insufficient Acc |
|---|---:|---:|---:|
| no relevance filter |  |  |  |
| length/filter only |  |  |  |
| overlap + NLI filter |  |  |  |

## 实验 5：hard negative 评估

| Subset | Rule F1 | Similarity-only F1 | NLI Attribution F1 |
|---|---:|---:|---:|
| same-topic different-location |  |  |  |
| same-person different-time |  |  |  |
| same-location different-event |  |  |  |
| high similarity OOC |  |  |  |
