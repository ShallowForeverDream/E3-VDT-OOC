# VDT-COVE-Attr v2 本地实验交接清单

## 已由远程完成

1. 新增代码模块：
   - `src/e3vdt/event/normalize.py`
   - `src/e3vdt/event/enhanced_extractor.py`
   - `src/e3vdt/attribution/field_nli_v2.py`
   - `src/e3vdt/attribution/evidence_relevance_v2.py`
   - `src/e3vdt/attribution/graph_alignment_v2.py`
2. 新增实验脚本：
   - `scripts/event/extract_event_tuples_v2.py`
   - `scripts/attribution/run_field_nli_attribution_v2.py`
   - `scripts/eval/evaluate_attribution_v2.py`
   - `scripts/run_vdt_cove_attr_v2_experiments.ps1`
   - `scripts/collect_v2_report_tables.py`
3. 新增完整技术路线文档：
   - `docs/VDT_COVE_ATTR_V2_IMPLEMENTATION_ROUTER.md`
4. 已填好 VDT baseline 表：
   - `bbc,guardian`: F1 0.7353 / Acc 0.7383 / AUC 0.7398
   - `usa_today,washington_post`: F1 0.8032 / Acc 0.8032 / AUC 0.8028

## 本地需要跑的实验

### 1. Smoke test：500 条

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
git pull
powershell -ExecutionPolicy Bypass -File .\scripts\run_vdt_cove_attr_v2_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -Python python `
  -MaxRecords 500 `
  -EvalSampleN 80 `
  -NliModel facebook/bart-large-mnli
```

如果 NLI 模型下载失败，先 smoke test：

```powershell
-NoTransformers
```

但正式结果必须尽量使用 NLI 模型。

### 2. 检查输出

```powershell
Get-Content outputs\input_check.json
Get-Content outputs\cove_lite_context_pairs.jsonl.stats.json
Get-Content outputs\event_tuples_v2.jsonl.stats.json
Get-Content outputs\field_nli_attribution_v2.jsonl.stats.json
Get-Content examples\attribution_eval_candidates.jsonl -TotalCount 5
Get-Content outputs\report_tables_v2.md
```

### 3. 人工标注

复制：

```powershell
Copy-Item examples\attribution_eval_candidates.jsonl examples\attribution_eval_set.jsonl -Force
```

人工填：

```json
{
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "annotator": "A",
  "rationale": "current caption and true context disagree on location",
  "annotation_status": "done"
}
```

允许字段：

```text
entity
location
time
event_type
relation
context_omission
evidence_insufficient
```

### 4. 人工标注后评测

```powershell
python scripts\eval\evaluate_attribution_v2.py `
  --gold examples\attribution_eval_set.jsonl `
  --pred outputs\field_nli_attribution_v2.jsonl `
  --output outputs\attribution_eval_v2_metrics.json

python scripts\collect_v2_report_tables.py --output outputs\report_tables_v2.md
```

### 5. 发回给我

```text
outputs/input_check.json
outputs/cove_lite_context_pairs.jsonl.stats.json
outputs/event_tuples_v2.jsonl.stats.json
outputs/field_nli_attribution_v2.jsonl.stats.json
outputs/attribution_eval_v2_metrics.json
outputs/report_tables_v2.md
```

## 报告中不能写的话

不要写：

```text
E3-VDT 主分类超过 VDT。
我们提出了真实归因数据集。
规则抽取就是高标准解释方法。
```

应该写：

```text
VDT 主分类已复现。
规则抽取只作为 baseline。
主方法是 COVE-lite + enhanced event extraction + field-wise NLI + evidence relevance。
归因效果通过人工 gold set 的 Type Acc / Field-F1 验证。
```
