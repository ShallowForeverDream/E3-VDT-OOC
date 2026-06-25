# VDT-COVE-Attr 模块方法深挖版（修正版：可控反事实归因训练路线）

> 本文档修正上一版“VDT-COVE-Attr 模块方法深挖版”的技术主线。  
> 旧版主要把 `COVE-lite + enhanced event extraction + field-wise NLI + graph alignment` 写成最终归因方法。  
> 经过讨论后，这个表述需要修正：由于 NewsCLIPpings 原始数据集只有 OOC / Non-OOC 二分类标签，没有细粒度错配原因标签，仅靠 NLI / graph 规则直接输出原因，缺少监督训练和强验证支撑。  
>
> 新版最终路线改为：  
> **VDT 主分类 + COVE-lite true context + Controlled Counterfactual Attribution Data + Attribution Head 监督训练 + 人工真实 OOC 评估。**
>
> 简言之：  
> **VDT 判断“是不是错配”；我们通过可控反事实样本训练模块判断“为什么错配”。**

---

## 0. 为什么要修正旧版路线

### 0.1 旧版路线的问题

旧版路线大致是：

```text
VDT baseline
  ↓
COVE-lite true context
  ↓
Enhanced event extraction
  ↓
Evidence relevance
  ↓
Field-wise NLI
  ↓
Graph alignment
  ↓
直接输出 mismatch_type / conflict_fields
```

这个链路可以作为 **zero-shot / weak attribution baseline**，但不能作为最终高标准主方法，原因有三点：

1. **NewsCLIPpings 原始数据没有错配原因标签**  
   原始标签只有 OOC / Non-OOC，不能直接验证 `location mismatch`、`time mismatch`、`entity mismatch` 是否正确。

2. **Field-wise NLI 和 graph alignment 是推理/规则链，不是监督训练模型**  
   它们可以提供特征、校验和 baseline，但如果没有 gold attribution labels，不能证明这些输出就是正确解释。

3. **真实 OOC 样本可能多字段同时错配**  
   原始 OOC pair 是图文重新配对得到的，可能同时发生人物、地点、时间、事件类型、语义关系多重冲突。若直接给它打一个单一错配原因，标签会很脏。

因此，新版必须解决一个核心问题：

> **没有细粒度归因标签时，如何构造可训练、可验证的错配原因标签？**

---

## 1. 新版最终技术路线总览

新版路线是：

```text
[A] VDT baseline classification
  image + current_caption -> OOC / Non-OOC

[B] COVE-lite true context construction
  image_id -> VisualNews original context
  得到 true_image_context

[C] Controlled counterfactual attribution data construction
  从 label=0 的 non-OOC 样本出发
  只替换 current_caption 中一个字段
  构造具有确定错配原因的负样本

[D] Feature construction
  event extraction
  field-wise NLI scores
  evidence relevance scores
  graph alignment features

[E] Attribution Head supervised training
  输入特征 -> mismatch_type + conflict_fields

[F] Evaluation
  synthetic held-out test
  manual real OOC attribution set
  hard negative set
```

最终一句话：

```text
我们不直接给原始 OOC 样本猜错配原因，而是从原本匹配的图文对出发，通过单字段反事实编辑构造带确定错配原因的训练样本，再训练 attribution head，并用人工真实 OOC 子集验证泛化能力。
```

---

## 2. 任务定义：系统最终做什么

### 2.1 输入

```json
{
  "image": "news image",
  "current_caption": "caption currently paired with this image"
}
```

### 2.2 VDT 输出

```json
{
  "vdt_label": "OOC",
  "vdt_score": 0.87
}
```

VDT 只回答：

```text
这张图和这段文字是否错配？
```

### 2.3 我们的归因模块输出

```json
{
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "explanation": "VDT 判断该图文对为 OOC。归因模块认为当前 caption 中的地点字段与图片真实语境冲突。"
}
```

归因模块回答：

```text
如果错配，主要错在哪里？
```

---

## 3. 数据问题：为什么不能直接监督训练

NewsCLIPpings 的原始构造目标是生成 out-of-context image-caption pairs。它通过重新配对未篡改图片和未篡改文本来构造 OOC 样本，强调的是“图片和文本本身都真实，但被错误配对”。因此它适合训练 OOC 二分类检测器，但不提供“这条样本到底是人物错配、地点错配、时间错配还是事件错配”的细粒度标签。

所以：

```text
原始 NewsCLIPpings:
  有：OOC / Non-OOC
  无：mismatch_type / conflict_fields
```

这意味着：

```text
不能直接把原始 OOC 样本作为 location/time/entity supervised training data。
```

我们必须自己构造细粒度标签。

---

## 4. 核心创新：Controlled Counterfactual Attribution Data

### 4.1 核心思想

只从原本匹配的 non-OOC 样本出发：

```text
image 与 caption 原本匹配
```

然后只改 caption 中一个事件字段，例如：

```text
只改地点
只改时间
只改人物/组织实体
```

这样构造出来的新 caption 与图片真实语境只在一个字段上冲突。于是错配原因由编辑操作天然决定。

示例：

```text
原始匹配样本：
Biden met officials in Washington in 2024.

只改地点：
Biden met officials in Paris in 2024.
```

构造标签：

```json
{
  "label": "OOC",
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "edit_type": "location_swap"
}
```

这就是 **controlled counterfactual negative**。

---

## 5. 反事实样本类别与数量设计

### 5.1 第一阶段：smoke test

先跑通流程，不追求规模。

| 类别 | 数量 |
|---|---:|
| none / positive | 200 |
| location mismatch | 200 |
| temporal mismatch | 200 |
| entity mismatch | 200 |

总量约：

```text
800 条
```

用途：

```text
验证字段抽取、span 定位、替换、校验、训练、评测全流程能跑通。
```

### 5.2 第二阶段：正式小规模实验

适合答辩后补实验。

| 类别 | Train | Val | Test |
|---|---:|---:|---:|
| none / positive | 3000 | 500 | 500 |
| location mismatch | 3000 | 500 | 500 |
| temporal mismatch | 3000 | 500 | 500 |
| entity mismatch | 3000 | 500 | 500 |

总量：

```text
Train 12000
Val 2000
Test 2000
```

### 5.3 第三阶段：扩大规模

如果指标还在提升，可以扩大到：

| 类别 | Train | Val | Test |
|---|---:|---:|---:|
| none / positive | 10000 | 1000 | 1000 |
| location mismatch | 10000 | 1000 | 1000 |
| temporal mismatch | 10000 | 1000 | 1000 |
| entity mismatch | 10000 | 1000 | 1000 |

总量：

```text
Train 40000
Val 4000
Test 4000
```

不建议一开始跑 126 万全量。原因：

```text
1. 构造质量比规模更重要；
2. 字段抽取和替换需要校验；
3. 学习曲线稳定后继续加数据收益有限；
4. 课程项目不需要全量训练才能答辩。
```

---

## 6. 为什么第一版只做 location / time / entity

| 类型 | 是否先做 | 原因 |
|---|---:|---|
| location mismatch | 做 | 地点 span 明确，替换可控 |
| temporal mismatch | 做 | 年份/日期 span 明确，旧图新用常见 |
| entity mismatch | 做 | 人物/组织可按同类型替换 |
| event-type mismatch | 暂缓 | 改事件类型容易改变整句语义 |
| relation mismatch | 暂缓 | 改谓词可能导致句子不自然 |
| context omission | 不做自动构造 | 很难定义唯一遗漏字段 |

第一版只做：

```text
none
location mismatch
temporal mismatch
entity mismatch
```

这样标签干净、构造稳定、评测简单。

---

## 7. 反事实样本构造的完整流程

### 7.1 Step 1：筛选 non-OOC 样本

输入：

```json
{
  "sample_id": "...",
  "image_id": "...",
  "current_caption": "Biden met officials in Washington in 2024.",
  "true_image_context": "Biden met officials in Washington in 2024.",
  "label": 0
}
```

只保留：

```text
label = 0
```

原因：

```text
只有原本匹配的样本，改一个字段后，错配原因才是确定的。
```

原本 OOC 样本不用于细粒度监督训练，因为它可能已经有多个冲突字段。

---

### 7.2 Step 2：抽取可编辑字段与 span

不是只抽字段值，还必须找到字段在文本中的位置。

例如：

```text
Biden met officials in Washington in 2024.
```

抽取：

```json
{
  "entities": [
    {"text": "Biden", "type": "PERSON", "start": 0, "end": 5}
  ],
  "locations": [
    {"text": "Washington", "type": "GPE", "start": 24, "end": 34}
  ],
  "times": [
    {"text": "2024", "type": "YEAR", "start": 38, "end": 42}
  ]
}
```

为什么必须要 span：

```text
因为我们要在原句上做最小编辑，只替换目标字段，不重写整句。
```

如果没有 span，就跳过该字段，不构造对应反事实样本。

实现方式：

```text
1. spaCy / Stanza / HanLP NER 提供实体 span；
2. 时间用正则提供 span；
3. 地点用 NER 的 GPE/LOC/FAC span；
4. 规则字段如果没有 span，则用字符串查找定位；
5. 多次出现同一字段时，只替换一个出现位置。
```

---

### 7.3 Step 3：建立 replacement pool

替换词不能手写几个固定词，而应从训练集自动收集。

#### location pool

从所有 non-OOC caption 里收集：

```text
Paris
London
Washington
Tokyo
Moscow
Gaza
Beijing
New York
...
```

并按类型区分：

```text
city
country
region
facility
```

第一版如果类型区分困难，至少要求：

```text
GPE/LOC/FAC 换 GPE/LOC/FAC
```

#### time pool

第一版优先只做年份：

```text
2018
2019
2020
2021
2022
2023
2024
```

替换规则：

```text
YEAR -> YEAR
```

暂不做复杂日期：

```text
Monday -> Friday
January -> March
June 5 -> July 9
```

因为复杂日期更容易引起语法和上下文问题。

#### entity pool

实体按类型分：

```text
PERSON: Biden, Trump, Macron, Sunak
ORG: UN, NATO, EU, WHO
NORP/GPE: American, French, China, United States
```

替换规则：

```text
PERSON -> PERSON
ORG -> ORG
NORP/GPE -> NORP/GPE
```

禁止：

```text
Biden -> Paris
UN -> 2024
Washington -> Trump
```

---

### 7.4 Step 4：单字段最小替换

对每条样本最多构造三种候选，但每个候选只改一个字段。

#### location_swap

原句：

```text
Biden met officials in Washington in 2024.
```

替换：

```text
Washington -> Paris
```

输出：

```text
Biden met officials in Paris in 2024.
```

标签：

```json
{
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "edit_type": "location_swap"
}
```

#### time_swap

原句：

```text
Biden met officials in Washington in 2024.
```

替换：

```text
2024 -> 2020
```

输出：

```text
Biden met officials in Washington in 2020.
```

标签：

```json
{
  "gold_mismatch_type": "temporal mismatch",
  "gold_conflict_fields": ["time"],
  "edit_type": "time_swap"
}
```

#### entity_swap

原句：

```text
Biden met officials in Washington in 2024.
```

替换：

```text
Biden -> Trump
```

输出：

```text
Trump met officials in Washington in 2024.
```

标签：

```json
{
  "gold_mismatch_type": "entity mismatch",
  "gold_conflict_fields": ["entity"],
  "edit_type": "entity_swap"
}
```

---

### 7.5 Step 5：构造后重新抽取并校验

反事实样本不能替换完就直接用，必须重新抽取校验。

对 edited_caption 和 true_image_context 重新运行 event extraction：

```text
edited_caption -> edited_event
true_image_context -> true_event
```

校验规则：

```text
1. target_field_changed = true
   目标字段确实变化。

2. other_fields_preserved = true
   非目标字段尽量保持一致。

3. replacement_type_valid = true
   替换词类型正确。

4. edited_caption_valid = true
   替换后文本非空，长度正常，没有明显乱码。

5. target_nli_not_entailment = true
   对目标字段构造 hypothesis 后，true context 不应 entail 新字段。

6. multi_conflict_risk = low
   不应同时引入多个非目标字段变化。
```

保留通过校验的样本。丢弃失败样本。

---

## 8. 输出数据格式

每条反事实样本：

```json
{
  "sample_id": "cf_000001",
  "source_sample_id": "...",
  "image_id": "...",

  "true_image_context": "Biden met officials in Washington in 2024.",
  "original_caption": "Biden met officials in Washington in 2024.",
  "edited_caption": "Biden met officials in Paris in 2024.",

  "label": 1,
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "edit_type": "location_swap",

  "edit_span": {
    "field": "location",
    "old": "Washington",
    "new": "Paris",
    "start": 24,
    "end": 34
  },

  "validation": {
    "target_field_changed": true,
    "other_fields_preserved": true,
    "replacement_type_valid": true,
    "edited_caption_valid": true,
    "target_nli_not_entailment": true
  }
}
```

正样本：

```json
{
  "sample_id": "pos_000001",
  "source_sample_id": "...",
  "image_id": "...",

  "true_image_context": "Biden met officials in Washington in 2024.",
  "original_caption": "Biden met officials in Washington in 2024.",
  "edited_caption": "Biden met officials in Washington in 2024.",

  "label": 0,
  "gold_mismatch_type": "none",
  "gold_conflict_fields": [],
  "edit_type": "none"
}
```

---

## 9. 原始 OOC 样本怎么处理

原始 OOC 样本不用于第一阶段细粒度监督训练。

它们可以标记为：

```json
{
  "label": 1,
  "gold_mismatch_type": "uncontrolled/global mismatch",
  "gold_conflict_fields": ["unknown"]
}
```

用途：

```text
1. VDT 二分类评估；
2. 人工真实 OOC 归因评测候选；
3. 检查 counterfactual-trained attribution head 是否能泛化到真实 OOC。
```

不能直接作为：

```text
location/time/entity supervised labels
```

除非人工标注。

---

## 10. Attribution Head 训练方案

### 10.1 为什么不重训 VDT

VDT 负责主分类，已经有复现结果。我们不修改 VDT，是为了：

```text
1. 保留已复现主分类性能；
2. 避免反事实合成数据污染 OOC 主分类；
3. 让分类和解释分开评估。
```

我们训练的是 VDT 后面的：

```text
Attribution Head
```

### 10.2 输入特征

每条样本经过 v2 pipeline 生成特征。

#### Field-wise NLI features

每个字段三个分数：

```text
entity_entailment
entity_neutral
entity_contradiction

location_entailment
location_neutral
location_contradiction

time_entailment
time_neutral
time_contradiction

event_type_entailment
event_type_neutral
event_type_contradiction

relation_entailment
relation_neutral
relation_contradiction
```

共 15 维。

#### Evidence relevance features

```text
evidence_relevance
text_similarity
entity_overlap
location_overlap
time_overlap
event_type_overlap
relation_overlap
filled_true_fields
filled_current_fields
context_length
```

#### Graph alignment features

```text
graph_alignment_score
num_current_edges
num_true_edges
has_relation_conflict
```

#### Optional VDT features

```text
vdt_score
vdt_label
```

第一版可以不使用 VDT score，因为反事实构造样本的核心目标是训练归因，不是重做主分类。

---

### 10.3 输出标签

任务一：mismatch_type 多分类。

第一版类别：

```text
none
location mismatch
temporal mismatch
entity mismatch
```

任务二：conflict_fields 多标签。

第一版字段：

```text
location
time
entity
```

---

### 10.4 模型结构

第一版推荐轻量 MLP：

```text
input features
  -> Linear
  -> ReLU
  -> Dropout
  -> Linear
  -> mismatch_type logits
  -> conflict_fields logits
```

损失函数：

```text
L = CE(mismatch_type) + λ * BCE(conflict_fields)
```

其中：

```text
CE：多分类交叉熵
BCE：多标签二元交叉熵
λ：两项损失权重，默认 1.0
```

也可以先做传统模型 baseline：

```text
Logistic Regression
Random Forest
MLP
```

报告中建议至少比较：

```text
Rule sidecar
Field-wise NLI
Attribution Head
```

---

## 11. 实验设计

### 11.1 反事实生成质量表

| Edit type | Generated | Kept | Keep rate |
|---|---:|---:|---:|
| location_swap | 待跑 | 待跑 | 待跑 |
| time_swap | 待跑 | 待跑 | 待跑 |
| entity_swap | 待跑 | 待跑 | 待跑 |

这张表证明：

```text
我们不是随便生成负样本，而是经过自动校验过滤。
```

### 11.2 Synthetic held-out test

在反事实 test set 上比较：

| Method | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |
|---|---:|---:|---:|---:|
| Majority | 待跑 | 待跑 | 待跑 | 待跑 |
| Rule sidecar | 待跑 | 待跑 | 待跑 | 待跑 |
| Field-wise NLI | 待跑 | 待跑 | 待跑 | 待跑 |
| Attribution Head | 待跑 | 待跑 | 待跑 | 待跑 |

作用：

```text
验证模型是否学会 controlled perturbation。
```

但注意：

```text
synthetic test 高分不等于真实 OOC 泛化好。
```

### 11.3 Manual real OOC attribution set

人工标 50–100 条真实 OOC 样本：

```text
gold_mismatch_type
gold_conflict_fields
rationale
```

比较：

| Method | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |
|---|---:|---:|---:|---:|
| Rule sidecar | 待跑 | 待跑 | 待跑 | 待跑 |
| Field-wise NLI | 待跑 | 待跑 | 待跑 | 待跑 |
| Counterfactual-trained Attribution Head | 待跑 | 待跑 | 待跑 | 待跑 |

这张表是最终最重要的。

### 11.4 Hard negative evaluation

构造或筛选：

```text
same-topic different-location
same-person different-time
same-location different-event
high-similarity but OOC
```

验证：

```text
模型是否只是学到了相似度 shortcut。
```

---

## 12. 消融实验

必须做：

| Variant | 说明 |
|---|---|
| Rule sidecar | 旧规则 baseline |
| Field-wise NLI only | 不训练，只用 NLI 阈值判断 |
| Attr Head w/o NLI | 不使用 NLI 特征 |
| Attr Head w/o evidence relevance | 去掉证据相关性特征 |
| Attr Head w/o graph features | 去掉图对齐特征 |
| Full Attr Head | 使用全部特征 |

目的：

```text
证明 NLI、evidence relevance、graph features 是否真的有贡献。
```

如果某个模块没贡献，要如实写。

---

## 13. 新版贡献点

### 13.1 贡献一：从无细粒度标签到可控归因监督

原始 NewsCLIPpings 无错配原因标签。我们通过 non-OOC 单字段反事实编辑构造带确定 conflict_fields 的训练样本。

### 13.2 贡献二：单字段最小编辑保证标签可解释

每个负样本只改变一个字段，因此标签由编辑操作决定：

```text
location_swap -> location mismatch
time_swap -> temporal mismatch
entity_swap -> entity mismatch
```

### 13.3 贡献三：VDT 主分类与归因模块解耦

VDT 不变，Attribution Head 只做解释。分类和解释分开评估。

### 13.4 贡献四：NLI / evidence / graph 作为可学习特征

旧版中它们是直接决策规则；新版中它们作为特征、校验器和 baseline，再通过监督训练学习组合。

### 13.5 贡献五：真实 OOC 人工集验证泛化

合成数据只用于训练，最终可靠性看人工真实 OOC attribution set。

---

## 14. 和旧版文档的关系

旧版内容哪些保留：

```text
COVE-lite true context construction 保留。
Enhanced event extraction 保留。
Field-wise NLI 保留。
Evidence relevance 保留。
Graph alignment-lite 保留。
```

旧版内容哪些修正：

```text
Field-wise NLI 不再是最终唯一主方法，而是 zero-shot baseline + feature generator。
Graph alignment 不再被包装成完整 Evidence-GNN，而是 graph feature / baseline。
规则抽取仍然只是 baseline。
最终主方法变为 counterfactual-trained Attribution Head。
```

旧版不准确的表述：

```text
COVE-lite + NLI + graph 直接完成可解释归因。
```

新版准确表述：

```text
COVE-lite + NLI + graph 提供可验证特征和 baseline；细粒度归因监督来自 controlled counterfactual data；最终由 Attribution Head 学习 mismatch_type / conflict_fields。
```

---

## 15. Codex 实现任务

### 15.1 新增反事实构造脚本

```text
scripts/data/build_controlled_counterfactuals.py
```

输入：

```text
outputs/cove_lite_context_pairs.jsonl
outputs/event_tuples_v2.jsonl
```

输出：

```text
outputs/counterfactual_attribution_train.jsonl
outputs/counterfactual_attribution_val.jsonl
outputs/counterfactual_attribution_test.jsonl
outputs/counterfactual_generation_stats.json
```

要求：

```text
1. 只从 label=0 的 non-OOC 样本构造；
2. 支持 location_swap、time_swap、entity_swap；
3. 每条负样本只改一个字段；
4. 字段必须有 span；
5. replacement 必须同类型；
6. 构造后重新抽取 event tuple 校验；
7. 输出 gold_mismatch_type 和 gold_conflict_fields；
8. 输出 keep rate 统计；
9. 保留 positive none 样本；
10. 支持 --max-per-type 控制每类数量。
```

### 15.2 新增特征构造脚本

```text
scripts/features/build_attribution_features.py
```

输入：

```text
counterfactual jsonl
field_nli_attribution_v2.jsonl
```

输出：

```text
outputs/attribution_features_train.csv
outputs/attribution_features_val.csv
outputs/attribution_features_test.csv
```

特征：

```text
field-wise NLI scores
evidence relevance scores
field overlaps
graph alignment features
```

### 15.3 新增训练脚本

```text
scripts/train/train_attribution_head.py
```

输入：

```text
attribution_features_train.csv
attribution_features_val.csv
attribution_features_test.csv
```

输出：

```text
outputs/attribution_head_model.pkl
outputs/attribution_head_metrics.json
outputs/attribution_head_metrics.csv
```

模型：

```text
Logistic Regression baseline
MLP attribution head
```

指标：

```text
Type Acc
Field Micro-F1
Field Macro-F1
Exact Match
```

---

## 16. 本地执行顺序

### 16.1 先跑已有 v2 链路

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

### 16.2 构造反事实样本

```powershell
python scripts\data\build_controlled_counterfactuals.py `
  --context-pairs outputs\cove_lite_context_pairs.jsonl `
  --event-tuples outputs\event_tuples_v2.jsonl `
  --out-dir outputs `
  --max-per-type 200
```

### 16.3 构造特征

```powershell
python scripts\features\build_attribution_features.py `
  --input outputs\counterfactual_attribution_train.jsonl `
  --output outputs\attribution_features_train.csv
```

### 16.4 训练 Attribution Head

```powershell
python scripts\train\train_attribution_head.py `
  --train outputs\attribution_features_train.csv `
  --val outputs\attribution_features_val.csv `
  --test outputs\attribution_features_test.csv `
  --output-dir outputs
```

---

## 17. 答辩口径

> 原始 NewsCLIPpings 只提供 OOC 二分类标签，不提供错配原因标签。因此我们不直接把原始 OOC 样本用于细粒度监督。我们的做法是从原本匹配的 non-OOC 样本出发，对 current caption 做单字段最小反事实编辑，例如只替换地点、时间或人物，从而构造具有确定错配原因的 controlled counterfactual negatives。由于每个负样本只改变一个事件字段，所以 conflict field 由编辑操作确定。随后我们用这些反事实样本训练 Attribution Head，并在人工标注的真实 OOC 子集上评估模型是否能泛化到真实错配场景。VDT 仍然负责 OOC 主分类，我们的模块只负责归因解释。

---

## 18. 最终版本一句话

```text
VDT-COVE-Attr 修正版不是简单用规则或 NLI 猜错配原因，而是通过 non-OOC 单字段反事实编辑构造带可控归因标签的数据，用 NLI/evidence/graph 特征训练 Attribution Head，再用人工真实 OOC 评测验证解释可靠性。
```

---

## 19. 参考方向

- NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media.
- COVE: COntext and VEracity prediction for out-of-context images.
- Counterfactually Augmented Data: Learning the Difference that Makes a Difference with Counterfactually-Augmented Data.
- MUSE / Similarity over Factuality: similarity shortcut in OOC detection.
- RED-DOT / CMIE / evidence relevance: evidence should be checked before attribution.
