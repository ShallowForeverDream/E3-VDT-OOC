# VDT-COVE-Attr 系统演示验收说明

本文件用于明天先答辩技术路线和系统演示。它区分“已经做成的系统闭环”和“答辩后继续补的严格实验”。

## 已经完成的系统闭环

| 模块 | 状态 | 文件/入口 |
|---|---|---|
| VDT 主分类复现 | completed | `examples/reproduction_metrics.json`, `docs/REPRODUCTION_STATUS.md` |
| COVE-lite true-context 输入 | completed for demo | `examples/cove_attr_demo_cases.jsonl` |
| Evidence relevance / sufficiency gate | completed for demo | `src/e3vdt/attribution/evidence_relevance.py` |
| Field-wise NLI-shaped attribution | completed for demo | `src/e3vdt/attribution/field_nli.py` |
| VDT-COVE-Attr 统一 pipeline | completed | `src/e3vdt/inference/cove_attr_pipeline.py` |
| Gradio 系统展示 | completed | `demo/app.py` 的 `VDT-COVE-Attr 主系统` tab |
| 系统演示自检 | completed | `python scripts/run_cove_attr_demo_cases.py` |

## 系统演示集结果

演示集是 curated smoke set，用于证明系统输入、归因输出、JSON schema 和 UI 一致，不替代最终大规模实验。

```powershell
python scripts/run_cove_attr_demo_cases.py
```

当前输出写入：

```text
examples/cove_attr_demo_outputs.json
```

## 明天答辩时可以说

> 我们已经把新技术路线做成一个可运行系统：VDT 负责主分类，COVE-lite 提供图片真实语境，Evidence relevance 判断证据是否足够，Field-wise NLI attribution 输出 entity/location/time/event_type/relation 字段矛盾。当前演示集用于验收系统闭环；归因模块是否真正可靠，将在答辩后用人工 gold set 和 ablation 实验验证。

## 明天答辩时不要说

- 不要说 curated demo 指标就是最终论文实验指标。
- 不要说当前 field-wise NLI 已经是训练好的 SOTA NLI 模型。
- 不要说归因模块已经在 NewsCLIPpings 全量上被严格证明。
- 不要说我们超过了 VDT 主分类性能。

## 答辩后必须补的实验

1. `context coverage`：确认 VisualNews metadata 路径和 image_id 对齐率。
2. `event extraction evaluation`：人工标注字段，评估 entity/location/time/event_type/relation F1。
3. `attribution baseline comparison`：majority/random/rule/similarity/NLI/evidence relevance。
4. `evidence relevance ablation`：有无 sufficiency gate 的归因差异。
5. `hard negative field-F1`：在 same-topic different-event 等样本上验证字段归因。
