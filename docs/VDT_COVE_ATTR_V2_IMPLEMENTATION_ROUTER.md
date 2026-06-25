# VDT-COVE-Attr v2 最终技术路线与实现 Router

## 0. 项目定位

本项目研究 **Out-of-Context Misinformation Detection with Attribution**：新闻图片本身可能真实，文本也可能来自真实新闻，但二者被错误配对，从而造成内容挪用和语境误导。

最终系统不是替代 VDT 主分类器，而是在 VDT 后面增加一个可验证的解释归因 sidecar：

```text
VDT baseline: image + current caption -> OOC / Non-OOC
VDT-COVE-Attr v2: OOC output + true image context -> mismatch_type + conflict_fields + explanation
```

核心原则：

```text
1. VDT 负责主分类。
2. Attribution sidecar 不覆盖 VDT label。
3. 解释模块必须有 true context / evidence grounding。
4. 解释是否可靠必须用人工 gold attribution set 评估。
5. 当前规则模块只作为 baseline，不作为最终主方法。
```

---

## 1. 数据与输入输出

### 1.1 原始数据

- **NewsCLIPpings**：提供 image-caption pair 和 OOC 二分类标签。
- **VisualNews metadata**：提供图片原始 caption / title / article metadata，可作为 `true_image_context`。

NewsCLIPpings 原始数据不提供：

```text
mismatch_type
conflict_fields
entity/location/time/event_type/relation labels
```

因此，任何归因标签都必须被称为 weak label 或人工标注子集，不可称为数据集原生标签。

### 1.2 系统输入

```json
{
  "sample_id": "...",
  "image_id": "...",
  "current_caption": "...",
  "vdt_label": "OOC",
  "vdt_score": 0.87
}
```

### 1.3 系统输出

```json
{
  "vdt_label": "OOC",
  "vdt_score": 0.87,
  "true_image_context": "VisualNews original caption/title",
  "current_event": {...},
  "true_event": {...},
  "evidence_relevance": {...},
  "field_nli": {...},
  "graph_alignment": {...},
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "explanation": "VDT detects OOC; true context contradicts the current caption on location."
}
```

---

## 2. 总体流程

```text
Step 0. VDT baseline reproduction
  - 已完成两组 strict BLIP-2/GaussianBlur baseline。

Step 1. COVE-lite true context construction
  - image_id -> VisualNews original caption/title/article metadata。

Step 2. Event tuple extraction
  - 从 current_caption 和 true_image_context 抽取 entity/location/time/event_type/relation。

Step 3. Evidence relevance / sufficiency
  - 判断 true_image_context 是否足以支持解释，避免无证据时强行输出 mismatch_type。

Step 4. Field-wise NLI contradiction detection
  - 对每个字段构造 hypothesis，用 NLI 判断 entailment/neutral/contradiction。

Step 5. Graph alignment
  - 对 OpenIE/SRL-like triples 做轻量图对齐，捕捉 subject-predicate-object 级冲突。

Step 6. Attribution decision
  - 融合 NLI conflict_fields、evidence_sufficiency、graph_conflicts 得到 mismatch_type。

Step 7. Evaluation
  - 人工 attribution gold set 上算 Type Acc / Field-F1 / Exact Match。
```

---

## 3. Step 1：COVE-lite true context construction

### 3.1 使用方法

脚本：

```text
scripts/context/build_cove_lite_context_pairs.py
```

输入：

```text
NewsCLIPpings data directory
VisualNews articles_metadata directory
```

输出：

```text
outputs/cove_lite_context_pairs.jsonl
outputs/cove_lite_context_pairs.jsonl.stats.json
```

每行：

```json
{
  "sample_id": "...",
  "image_id": "...",
  "current_caption": "...",
  "true_image_context": "...",
  "label": 1
}
```

### 3.2 方法来源

迁移 COVE 的 context-first 思路：OOC 解释先需要得到图像真实上下文，再用真实上下文判断当前 caption 是否成立。我们不完整复现 COVE 的检索/LLM 流程，而是利用 NewsCLIPpings 与 VisualNews 的数据关系直接得到 true context。

### 3.3 实验指标

```text
total
kept
coverage = kept / total
missing_text
missing_true_context
```

---

## 4. Step 2：事件字段抽取

### 4.1 当前规则 baseline

旧代码：

```text
src/e3vdt/event/extractor.py
```

性质：

```text
regex + keyword + small gazetteer
```

只能作为 rule baseline。

### 4.2 v2 enhanced extractor

新代码：

```text
src/e3vdt/event/enhanced_extractor.py
scripts/event/extract_event_tuples_v2.py
```

实现：

```text
1. 保留 rule extractor 作为 fallback。
2. 可选 spaCy NER：PERSON/ORG/NORP -> entities；GPE/LOC/FAC -> locations；DATE/TIME -> times。
3. 本地 OpenIE-like fallback triples：抽取常见新闻三元组 subject-predicate-object。
4. 归一化 location/event_type aliases。
```

输出：

```json
{
  "entities": [],
  "locations": [],
  "times": [],
  "event_types": [],
  "relations": [],
  "relations_structured": [
    {"subject": "protesters", "predicate": "gathered", "object": "Paris"}
  ]
}
```

### 4.3 为什么这样做

目标不是声称本地 fallback 是完整 OpenIE，而是建立可插拔抽取接口：

```text
rule baseline -> enhanced NER/OpenIE -> optional LLM JSON extractor
```

答辩中需要说明：当前 v2 已把规则降级为 baseline，并为后续接 spaCy/Stanza/HanLP/OpenIE/LLM 保留接口。

### 4.4 实验

人工标注 100-300 条 event tuple，比较：

```text
Rule extractor
Enhanced extractor
LLM JSON extractor（可选）
```

指标：

```text
Entity F1
Location F1
Time F1
Event Type F1
Relation F1
Exact Tuple Match
```

---

## 5. Step 3：Evidence relevance / sufficiency

代码：

```text
src/e3vdt/attribution/evidence_relevance_v2.py
```

输入：

```text
current_caption
true_image_context
current_event
true_event
```

计算：

```text
semantic text similarity
entity/location/time/event_type/relation overlaps
true_context length
filled true event field count
```

输出：

```json
{
  "evidence_relevance": 0.78,
  "evidence_sufficiency": "sufficient",
  "evidence_reason": "sufficient"
}
```

如果 evidence insufficient，后续不强行输出 location/time/entity mismatch，而是：

```text
uncertain / evidence insufficient
```

方法来源：RED-DOT 与 CMIE 共同说明 OOC 中外部证据不能默认可靠，必须进行相关性/充分性筛选。

实验：

```text
No relevance filter
Length-only filter
Overlap filter
NLI/overlap relevance filter
```

指标：

```text
Field-F1
Evidence-insufficient accuracy
错误强行归因数量
```

---

## 6. Step 4：Field-wise NLI contradiction detection

代码：

```text
src/e3vdt/attribution/field_nli_v2.py
scripts/attribution/run_field_nli_attribution_v2.py
```

对每个字段构造 hypothesis：

```text
entity: The image event involves X.
location: The image event took place in X.
time: The image event happened at X.
event_type: The image event is about X.
relation: The image event includes the relation or action: X.
```

premise：

```text
true_image_context
```

NLI 输出：

```json
{
  "location": {
    "label": "contradiction",
    "scores": {"entailment": 0.03, "neutral": 0.12, "contradiction": 0.85}
  }
}
```

判定：

```text
contradiction >= threshold -> conflict field
entailment >= threshold -> consistent
otherwise -> neutral / uncertain
```

这一步把原先字符串相似度规则升级为字段级语义矛盾检测。NLI 失败或未安装模型时，代码会降级到 lexical fallback，但实验报告必须标注使用的是 `facebook/bart-large-mnli` 还是 fallback。

---

## 7. Step 5：Graph alignment

代码：

```text
src/e3vdt/attribution/graph_alignment_v2.py
```

输入：

```text
current_event.relations_structured
true_event.relations_structured
```

实现：

```text
1. 对 subject/predicate/object 三元组做相似度对齐。
2. 若 subject 与 predicate 对齐但 object 差异大，则标记 relation conflict。
3. 输出 graph_alignment_score 和 graph_conflicts。
```

这不是完整 Evidence-GNN，而是 Evidence-GNN/EXCLAIM 思路的轻量可运行适配：先做 graph matching，后续可替换为 GNN。

---

## 8. Step 6：Attribution decision

脚本：

```text
scripts/attribution/run_field_nli_attribution_v2.py
```

决策逻辑：

```text
if evidence_sufficiency == insufficient:
    mismatch_type = uncertain / evidence insufficient
    conflict_fields = [evidence_insufficient]

elif field_nli has contradiction:
    conflict_fields = all contradictory fields
    mismatch_type = field with highest contradiction score

elif graph_alignment has conflict:
    conflict_fields += graph_conflicts
    mismatch_type = mapped graph conflict

else:
    mismatch_type = benign illustrative image / context omission
```

输出：

```text
outputs/field_nli_attribution_v2.jsonl
outputs/field_nli_attribution_v2.jsonl.stats.json
```

---

## 9. Evaluation

### 9.1 自动可生成表

脚本：

```text
scripts/collect_v2_report_tables.py
```

输出：

```text
outputs/report_tables_v2.md
```

已固定填入 VDT baseline 结果：

| Method | Target domain | Batch size | F1 | Acc | AUC | Status |
|---|---|---:|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur | bbc,guardian | 128 | 0.7353 | 0.7383 | 0.7398 | completed |
| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 128 | - | - | - | failed: CUDA OOM |
| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 64 | 0.8032 | 0.8032 | 0.8028 | completed |

### 9.2 需要人工标注后才能完成的表

脚本：

```text
scripts/eval/evaluate_attribution_v2.py
```

输入：

```text
examples/attribution_eval_set.jsonl
outputs/field_nli_attribution_v2.jsonl
```

输出：

```text
outputs/attribution_eval_v2_metrics.json
outputs/attribution_eval_v2_metrics.csv
```

指标：

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

---

## 10. 一键本地实验

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vdt_cove_attr_v2_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -Python python `
  -MaxRecords 500 `
  -EvalSampleN 80 `
  -NliModel facebook/bart-large-mnli
```

若模型下载或显存有问题：

```powershell
-NoTransformers
```

但报告中必须写明 fallback 仅用于 smoke test，正式结果应使用 NLI 模型。

---

## 11. 不能由远程直接完成的实验

由于实验依赖你本地 D/E 盘数据和 NLI 模型下载，远程已完成代码与脚本，但以下结果需要本地跑：

```text
1. COVE-lite coverage 表。
2. v2 event extraction coverage 表。
3. field-wise NLI attribution distribution 表。
4. 人工 gold attribution set 上的 Type Acc / Field-F1。
5. hard negative 子集上的 Field-F1。
```

这些表会由 `outputs/report_tables_v2.md` 自动汇总。

---

## 12. 答辩口径

我们不再把经验规则包装成创新。规则模块只作为 baseline。最终技术路线是：

```text
VDT baseline
+ COVE-lite true context
+ enhanced event extraction
+ evidence relevance / sufficiency
+ field-wise NLI contradiction
+ lightweight graph alignment
+ AMG-style manual attribution evaluation
```

一句话版本：

> 我们首先复现 VDT 作为 OOC 主分类 baseline。VDT 只能输出 OOC/Non-OOC，不能解释错在哪里。为此我们迁移 COVE 的 true-context 思路，用 VisualNews 原始上下文构造图像真实语境；再从 current caption 和 true context 中抽取事件字段；用 field-wise NLI 判断 entity、location、time、event_type、relation 是否存在 contradiction；结合 evidence relevance 避免证据不足时强行解释；最后用人工 attribution gold set 评估 mismatch type accuracy 和 conflict field F1。
