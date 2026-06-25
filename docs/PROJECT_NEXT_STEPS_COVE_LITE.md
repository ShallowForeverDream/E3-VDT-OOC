# 项目下一阶段：从规则 Demo 升级到 COVE-lite 归因评测

本页是当前项目后续工作的执行清单。项目不再把规则解释 demo 当成最终创新，而是转向一个可以被实验验证的方向：

> VDT 做 OOC / Non-OOC 主分类；COVE-lite true-context attribution 做错配字段归因。

## 1. 当前状态

已完成：

- VDT strict BLIP-2/GaussianBlur 两组核心复现；
- Gradio demo；
- event sidecar heuristic；
- accuracy-preserving 策略。

仍需补齐：

- COVE-lite true image context 数据构造；
- 弱归因标签生成；
- 人工标注归因评测集；
- majority / sampled / text-only / COVE-lite rule 对比；
- field-level F1 和 type accuracy。

## 2. 为什么要改

当前 demo 能输出错配类型，但这些输出主要来自规则。规则输出不能直接证明解释正确。为了避免答辩时被问住，必须补一个小规模人工评测集，证明该归因模块至少优于简单 baseline。

## 3. 新主线

```text
NewsCLIPpings current caption
+ VisualNews original context
-> event tuple extraction
-> event-field consistency
-> mismatch_type / conflict_fields
-> attribution evaluation
```

这相当于一个轻量版 COVE：不使用大模型预测图像上下文，而是直接利用 VisualNews 原始上下文作为图像真实语境。

## 4. 本地运行命令

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
git pull

.\scripts\run_cove_lite_attribution_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\metadata `
  -Python python `
  -EvalSampleN 120
```

如果 metadata 路径不对，先在 E 盘搜索：

```powershell
Get-ChildItem E:\OOC_Datasets\VisualNews -Recurse -File -Include *.p,*.pkl,*.pickle | Select-Object FullName, Length
```

把包含 `processed_*.p` 或 VisualNews article metadata 的目录填给 `-VisualNewsMetadataDir`。

## 5. 生成文件

```text
outputs/cove_lite_context_pairs.jsonl
outputs/weak_attribution_labels.jsonl
examples/attribution_eval_candidates.jsonl
```

人工把候选集标注为：

```text
examples/attribution_eval_set.jsonl
```

然后运行：

```powershell
python scripts/eval/run_attribution_baselines.py `
  --gold examples\attribution_eval_set.jsonl `
  --weak-labels outputs\weak_attribution_labels.jsonl `
  --output outputs\attribution_eval_metrics.json
```

## 6. 人工标注要求

每条样本至少填写：

```json
{
  "sample_id": "...",
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "annotator": "A",
  "rationale": "current caption and true context disagree on location",
  "annotation_status": "done"
}
```

字段集合：

```text
entity
location
time
event_type
relation
context_omission
evidence_insufficient
```

建议先标 100 条；时间不够则标 50 条，并在报告中说明限制。

## 7. 最终报告口径

可以说：

> 我们完成了 VDT baseline 复现，并在其上实现 COVE-lite true-context attribution。该模块不改变 VDT 主分类结果，而是输出错配类型和冲突字段，并通过人工归因评测集进行验证。

不要说：

> 我们已经做出超过 VDT 的主分类模型。
> 我们完成了完整 Evidence Gate 和 Event-Guided TTT。
> 弱标签不用人工验证就一定正确。
