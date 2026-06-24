# 系统输入输出接口

所有模型、demo、脚本都必须对齐这个 schema，避免系统和理论模块互相重写。

## 输入

```json
{
  "image": "path/to/image.jpg",
  "text": "A protest erupted in Paris on Monday.",
  "image_context": "People gathered in London in 2020.",
  "evidence": [{"source": "retrieved-caption", "text": "...", "score": 0.82}]
}
```

## 输出

```json
{
  "label": "OOC",
  "confidence": 0.87,
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location", "time"],
  "event_scores": {"entity": 0.82, "location": 0.21, "time": 0.35, "event_type": 0.76, "relation": 0.58},
  "text_event": {},
  "image_event": {},
  "evidence": [],
  "explanation": "图文主题相近，但地点字段不一致。",
  "model_version": "e3-vdt-demo-heuristic-v0.1",
  "classification_policy": "baseline_preserving",
  "decision_source": "vdt_baseline",
  "baseline_label": "OOC",
  "baseline_score": 0.87,
  "warnings": []
}
```

主标签：OOC / Non-OOC / Uncertain。

## Accuracy-preserving 约束

正式实验/论文口径默认使用 `classification_policy="baseline_preserving"`：

```text
label = baseline_label
confidence = baseline_score
```

事件字段模块只更新 `mismatch_type`、`conflict_fields`、`event_scores`、`explanation`，不覆盖主分类。因此 E3-VDT sidecar 模式的分类 Accuracy/F1 与 VDT baseline 持平；只有验证集证明不降指标时，才允许启用融合分类。
