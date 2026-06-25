# VDT-COVE-Attr v2 技术路线与实验包

本包用于明天技术路线答辩与后续本地实验。核心口径：

> VDT 负责 OOC / Non-OOC 主分类；VDT-COVE-Attr v2 在 VDT 后增加可验证的上下文归因模块。该模块借鉴 COVE 的 true-context 思路、RED-DOT/CMIE 的 evidence relevance 思路、MUSE 的 shortcut 反证设置、AMG 的 attribution evaluation 思路，用 field-wise NLI/事件图对齐判断图片被如何误用。

## 目录

- `docs/TECHNICAL_ROUTE.md`：最终技术路线。
- `docs/DEFENSE_SCRIPT.md`：答辩口径与问答。
- `docs/EXPERIMENT_PLAN.md`：实验计划与结果表模板。
- `docs/ANNOTATION_GUIDELINE.md`：人工归因标注规范。
- `scripts/run_all_local.ps1`：本地一键实验脚本。
- `scripts/00_check_inputs.py`：检查 NewsCLIPpings 与 VisualNews metadata 是否可读。
- `scripts/01_build_context_pairs.py`：构造 COVE-lite context pairs。
- `scripts/02_extract_events.py`：事件字段抽取，支持 rule / optional spaCy。
- `scripts/03_field_nli_attribution.py`：field-wise NLI 归因。
- `scripts/04_build_annotation_candidates.py`：抽取人工标注候选样本。
- `scripts/05_eval_attribution.py`：评估归因方法。
- `scripts/06_collect_tables.py`：收集结果表。
- `templates/attribution_eval_set.template.jsonl`：人工标注模板。

## 快速运行

在项目根目录 `D:\MY_PROJECT\OOC\E3-VDT-OOC` 解压本包，然后运行：

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
powershell -ExecutionPolicy Bypass -File .\vdt_cove_attr_package\scripts\run_all_local.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -Python python `
  -MaxRecords 500 `
  -NliModel facebook/bart-large-mnli
```

先跑 `MaxRecords 500`，确认输出非空后再跑全量。

## 关键产物

- `outputs/cove_lite_context_pairs.jsonl`
- `outputs/event_tuples.jsonl`
- `outputs/field_nli_attribution.jsonl`
- `examples/attribution_eval_candidates.jsonl`
- `outputs/attribution_eval_metrics.json`
- `outputs/report_tables.md`

## 明天答辩口径

明天先答技术路线，不承诺所有实验已完成。明确说：

1. 已复现 VDT 两组 strict BLIP-2/GaussianBlur baseline。
2. VDT 只能输出 OOC / Non-OOC，不解释错配原因。
3. 我们将可解释性拆成两个可验证子任务：事件抽取与上下文归因。
4. 当前规则 sidecar 只作为 baseline，最终主方法是 COVE-lite true context + field-wise NLI + evidence relevance + 人工归因评测。
5. 答辩后补齐人工 gold set 与 field-F1 / type accuracy 实验。
