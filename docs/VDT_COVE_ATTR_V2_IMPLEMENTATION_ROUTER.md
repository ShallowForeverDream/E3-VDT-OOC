# VDT-COVE-Attr v2 最终技术路线与实现 Router（详细版）

本文档用于报告、答辩和代码实现对齐。目标不是写模块名称，而是把每一步的 **输入是什么、经过什么操作、输出成什么、为什么要这样做、依据哪类已有工作、需要什么实验验证** 说清楚。

---

## 0. 项目定位：我们到底解决什么问题

本项目研究 **Out-of-Context Misinformation Detection with Attribution**。新闻图片本身可能真实，文本也可能来自真实新闻，但二者被错误配对后会造成内容挪用和语境误导。例如：一张旧新闻图片被放到新的战争、抗议、灾害或政治新闻中，图片和文字表面相关，但并不是同一事件。

我们不替代 VDT 主分类器。VDT 负责判断：

```text
image + current_caption -> OOC / Non-OOC
```

我们新增的是后置归因模块：

```text
VDT output + true_image_context -> mismatch_type + conflict_fields + field-level explanation
```

最终系统必须坚持五个原则：

```text
1. VDT 负责主分类。
2. Attribution sidecar 不覆盖 VDT label。
3. 解释必须基于 true context / evidence grounding。
4. 当前规则模块只作为 baseline，不作为最终主方法。
5. 归因是否可靠必须用人工 gold attribution set 评估。
```

---

## 1. 整体数据流：一个样本如何被处理

### 1.1 原始输入

一个 NewsCLIPpings 样本包含：

```json
{
  "sample_id": "...",
  "image_id": "...",
  "current_caption": "The caption currently paired with the image.",
  "label": 0 或 1
}
```

其中 `label` 只是 OOC 二分类标签：

```text
0 = Non-OOC / 图文真实匹配
1 = OOC / 图文被错误配对
```

NewsCLIPpings 原始数据不包含：

```text
mismatch_type
conflict_fields
entity/location/time/event_type/relation 归因标签
```

所以我们不能说“NewsCLIPpings 自带归因标签”。

### 1.2 COVE-lite true context 构造

NewsCLIPpings 来自 VisualNews。VisualNews metadata 中保存图片原始新闻上下文，例如原始 caption、article title、source metadata。我们利用 `image_id` 回查 VisualNews，构造：

```json
{
  "current_caption": "现在图文对里的文本",
  "true_image_context": "这张图片在 VisualNews 中原本对应的 caption/title/article metadata"
}
```

这一点借鉴 COVE 的 context-first 思路。COVE 的核心是：先得到图像真实上下文，再判断当前 caption 是否与该上下文一致。我们不完整复现 COVE 的检索和大模型流程，而是利用 NewsCLIPpings 和 VisualNews 的数据关系直接得到 true context。

### 1.3 v2 attribution 数据流

完整数据流是：

```text
NewsCLIPpings sample
  -> VDT baseline prediction
  -> COVE-lite true_image_context
  -> current_caption event extraction
  -> true_image_context event extraction
  -> evidence relevance / sufficiency scoring
  -> field-wise NLI contradiction detection
  -> lightweight graph alignment
  -> attribution decision
  -> mismatch_type + conflict_fields + explanation
```

---

## 2. Step 1：VDT baseline，为什么先保留不改

VDT 已经解决 OOC 主分类：

```text
image + current_caption -> OOC / Non-OOC
```

它的问题是：

```text
只输出是否 OOC，不输出错在哪里。
```

因此我们把 VDT 作为 frozen baseline，而不是一开始就修改 VDT encoder 或 classifier。这样有三个好处：

```text
1. 已复现的 VDT 指标可以稳定保留。
2. Attribution 模块失败不会污染主分类结果。
3. 实验可以分开评估：分类看 VDT，解释看 attribution。
```

已完成 VDT strict BLIP-2/GaussianBlur 结果：

| Method | Target domain | Batch size | F1 | Acc | AUC | Status |
|---|---|---:|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur | bbc,guardian | 128 | 0.7353 | 0.7383 | 0.7398 | completed |
| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 128 | - | - | - | failed: CUDA OOM |
| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 64 | 0.8032 | 0.8032 | 0.8028 | completed |

---

## 3. Step 2：事件字段抽取，逐步讲清楚

### 3.1 这一步要解决什么

我们要把自然语言文本变成结构化事件表示。否则后面无法判断“错在哪里”。

输入是两段文本：

```text
current_caption: 当前图文对里的文本
true_image_context: 图片真实上下文文本
```

目标输出是两个 event tuple：

```json
{
  "entities": [],
  "locations": [],
  "times": [],
  "event_types": [],
  "relations": [],
  "relations_structured": []
}
```

每个字段的含义：

```text
entities: 人物、组织、国家、机构、主体
locations: 地点、国家、城市、地区、机构地点
times: 年份、日期、星期、月份、相对时间
event_types: 事件类型，如 protest、war/conflict、disaster、politics、sports
relations: 行为关系标签，如 attack、meet、arrest、rescue
relations_structured: subject-predicate-object 三元组
```

示例：

```text
current_caption = "Protesters gathered in Paris on Monday."
```

抽取后：

```json
{
  "entities": ["Protesters"],
  "locations": ["Paris"],
  "times": ["Monday"],
  "event_types": ["protest"],
  "relations": ["gather"],
  "relations_structured": [
    {"subject": "Protesters", "predicate": "gathered", "object": "Paris"}
  ]
}
```

这一步的效果是：把原始文本从“不可比较的一整句话”变成“可逐字段比较的事件结构”。

---

### 3.2 为什么当前规则不能作为最终方法

旧版代码主要靠：

```text
正则表达式 + 关键词表 + 小型地点词表 + 字符串相似度
```

这只能作为 baseline。它的问题是：

```text
1. 地点覆盖不全。
2. 事件类型只靠关键词，容易漏检和误检。
3. 实体只靠大写短语，新闻实体抽取不稳定。
4. relation 只是少量关键词，不是真正关系抽取。
5. 无法识别 Biden visited Ukraine 与 Biden spoke in Washington 这种关系级错配。
```

因此 v2 将旧规则降级为 `Rule Extractor Baseline`，主路线改成：

```text
rule fallback
+ optional pretrained NER
+ time/location/event normalization
+ OpenIE/SRL-like triples
+ optional LLM JSON extractor
```

---

### 3.3 v2 enhanced extractor 的完整流程

v2 extractor 的代码入口：

```text
src/e3vdt/event/enhanced_extractor.py
scripts/event/extract_event_tuples_v2.py
```

对每一段文本，它按下面流程处理。

---

#### Step 2.1 文本清洗

输入：

```text
"  Protesters   gathered in Paris on Monday.  "
```

操作：

```text
1. 去除首尾空白；
2. 合并多个空格；
3. 保留原始大小写用于 NER；
4. 生成小写标准化版本用于匹配和归一化。
```

输出：

```text
"Protesters gathered in Paris on Monday."
```

为什么要做：

```text
清洗后可以避免空格、标点、大小写导致字段重复或匹配失败。
```

---

#### Step 2.2 Rule fallback 抽取基础字段

输入：清洗后的文本。

操作：调用旧规则抽取器：

```text
extract_event_tuple(text)
```

它会执行：

```text
1. TIME_RE 正则抽取时间；
2. KNOWN_LOCATIONS + 中文地点后缀正则抽取地点；
3. EVENT_KEYWORDS 抽取事件类型；
4. RELATION_KEYWORDS 抽取关系标签；
5. 英文大写短语 + 中文实体后缀抽取实体。
```

输出初始 event tuple：

```json
{
  "entities": ["Protesters"],
  "locations": ["Paris"],
  "times": ["Monday"],
  "event_types": ["protest"],
  "relations": ["gather"]
}
```

为什么仍保留规则：

```text
1. 无依赖，可复现；
2. 在模型不可用时作为 fallback；
3. 作为实验 baseline，证明 v2 方法是否真的优于规则。
```

但报告中必须说清楚：

```text
Rule fallback 不是最终高标准方法，只是 baseline 和兜底模块。
```

---

#### Step 2.3 预训练 NER 抽取实体、地点、时间

输入：同一段文本。

推荐工具：

```text
英文：spaCy / Stanza
中文：HanLP / Stanza 中文模型
```

v2 当前代码提供 spaCy 接口，如果本地安装 `en_core_web_sm`，则运行：

```text
spaCy(text) -> doc.ents
```

然后映射：

```text
PERSON / ORG / NORP -> entities
GPE / LOC / FAC -> locations
DATE / TIME -> times
```

示例：

```text
"US President Biden met officials in Washington in 2024."
```

spaCy 可能抽到：

```text
Biden -> PERSON
Washington -> GPE
2024 -> DATE
US -> GPE/NORP
```

映射后：

```json
{
  "entities": ["Biden"],
  "locations": ["Washington", "US"],
  "times": ["2024"]
}
```

为什么需要这一步：

```text
规则只能匹配有限词表，NER 可以识别词表之外的人名、组织、地名和日期。
```

论文/技术依据：

```text
命名实体识别是信息抽取基础任务，新闻事件抽取通常首先依赖 NER 识别事件论元，如人物、组织、地点、时间。
```

这一步在我们系统中起到的效果：

```text
提高 entity/location/time 字段覆盖率，降低规则漏检导致的 evidence_insufficient 或错误 mismatch。
```

---

#### Step 2.4 时间字段归一化

输入：

```text
["Monday", "June 2024", "2024", "6月25日"]
```

操作：

```text
1. 保留原始时间表达；
2. 可选用 dateparser/SUTime/HeidelTime 转成标准时间；
3. 对无法标准化的表达标记为 surface time。
```

输出：

```json
{
  "times": ["2024", "June 2024"],
  "time_norm": ["2024", "2024-06"]
}
```

为什么需要：

```text
OOC 中最常见误用之一是旧图新用。若时间字段不能归一化，就无法判断 temporal mismatch。
```

当前 v2 代码保留基础时间字段，后续可接 dateparser/SUTime。报告中要区分：

```text
当前可运行版本：基础时间抽取；
增强版本：时间标准化解析器。
```

---

#### Step 2.5 地点字段归一化

输入：

```text
["US", "U.S.", "USA", "United States", "Washington D.C."]
```

操作：

```text
1. 小写化；
2. 去标点；
3. alias dictionary 映射；
4. 可选接 GeoNames / gazetteer 做标准化。
```

示例：

```text
US / U.S. / USA / America -> united states
Washington D.C. -> washington
UK / Britain / England -> united kingdom
```

输出：

```json
{
  "locations": ["united states", "washington"]
}
```

为什么需要：

```text
字符串不同不等于地点不同。如果不归一化，US 与 United States 可能被误判为 location mismatch。
```

这一步起到的效果：

```text
减少同义地名导致的假冲突。
```

---

#### Step 2.6 事件类型归一化

输入：

```text
["war", "attack", "strike", "conflict"]
```

操作：

```text
把同义或近义事件归到统一 taxonomy。
```

示例：

```text
war / attack / strike / conflict -> war/conflict
rally / demonstration / march -> protest
election / vote -> politics
trial / arrest -> court/crime
```

输出：

```json
{
  "event_types": ["war/conflict"]
}
```

为什么需要：

```text
事件类型是粗粒度语义字段，不应被表面词差异影响。
```

这一步使 event_type 比较不再完全依赖关键词原词。

---

#### Step 2.7 OpenIE / SRL-like 关系三元组抽取

只抽字段还不够，因为很多误用发生在关系层面：同一人物、同一地点，但行为关系不同。

目标是把句子转成：

```text
subject - predicate - object
```

例如：

```text
"Protesters gathered in Paris."
```

转成：

```json
{
  "subject": "Protesters",
  "predicate": "gathered",
  "object": "Paris"
}
```

再如：

```text
"Police arrested demonstrators near the court."
```

转成：

```json
{
  "subject": "Police",
  "predicate": "arrested",
  "object": "demonstrators near the court"
}
```

论文/技术依据：

```text
Open Information Extraction 和 Semantic Role Labeling 的目标就是从自然语言中抽取机器可处理的关系结构或谓词-论元结构。事件级 OOC 解释不能只依赖关键词，需要知道谁对谁做了什么。
```

AMR/neural-symbolic OOC 相关工作也说明，把 caption 转成结构化 fact queries 或语义图后，可以更清楚地定位 cross-modal contradictions。

当前 v2 代码实现的是 `OpenIE-like fallback triples`，它不是完整 OpenIE，但已经建立了接口：

```text
relations_structured = [
  {subject, predicate, object}
]
```

后续如果安装完整 OpenIE/SRL/AMR，可以直接替换该字段来源。

这一步起到的效果：

```text
从字段集合比较升级到事件关系比较，为 graph alignment 提供输入。
```

---

#### Step 2.8 合并去重与输出 EventTuple

现在有三类来源：

```text
1. rule fallback 输出；
2. NER 输出；
3. OpenIE/SRL-like triples 输出。
```

合并操作：

```text
1. 对每个字段去空白；
2. 统一小写或标准形式；
3. 去重；
4. location / event_type 做 alias normalization；
5. relation 既保留标签，也保留 structured triples。
```

最终输出：

```json
{
  "entities": ["biden"],
  "locations": ["washington"],
  "times": ["2024"],
  "event_types": ["politics"],
  "relations": ["meet"],
  "relations_structured": [
    {"subject": "Biden", "predicate": "met", "object": "officials"}
  ],
  "extractor": "enhanced"
}
```

这就是后续 NLI 和 graph alignment 的输入。

---

## 4. Step 3：Evidence relevance / sufficiency，为什么需要

### 4.1 问题

不是所有 true context 都足以解释错配。VisualNews 中有些上下文可能很短：

```text
"A file photo from Reuters."
```

这类上下文没有人物、地点、时间或事件类型。此时不能强行判断：

```text
location mismatch
```

而应输出：

```text
evidence insufficient
```

### 4.2 方法依据

RED-DOT 指出，外部证据不能默认相关，必须进行 relevant evidence detection。CMIE 也指出噪声证据会损害 OOC 检测，因而需要 association / relevance scoring。

我们迁移的是思想：

```text
在做归因前先判断证据是否相关、是否充分。
```

### 4.3 我们怎么计算

输入：

```text
current_caption
true_image_context
current_event
true_event
```

计算四组特征：

```text
1. text_similarity(current_caption, true_image_context)
2. entity/location/time/event_type/relation overlap
3. true_context length
4. true_event 中非空字段数量
```

组合成：

```text
evidence_relevance =
  0.35 * text_similarity
+ 0.35 * field_overlap_mean
+ 0.15 * context_len_score
+ 0.15 * information_score
```

输出：

```json
{
  "evidence_relevance": 0.78,
  "evidence_sufficiency": "sufficient",
  "evidence_reason": "sufficient"
}
```

如果：

```text
true_context 太短
或 true_event 没抽到字段
或 relevance 低于阈值
```

则输出：

```json
{
  "evidence_sufficiency": "insufficient",
  "evidence_reason": "true_context_too_short / no_fields / low_relevance"
}
```

### 4.4 起到的效果

它使系统具备拒答能力：

```text
证据不足时不强行解释。
```

这比旧规则更符合事实核验系统要求。

---

## 5. Step 4：Field-wise NLI contradiction detection，具体怎么做

### 5.1 为什么用 NLI

旧方法用字符串相似度：

```text
Paris vs London -> low similarity -> conflict
US vs United States -> low/medium similarity -> 可能误判
```

NLI 判断的是语义关系：

```text
entailment: true context 支持当前字段声明
neutral: 无法判断
contradiction: true context 与当前字段声明矛盾
```

所以 field-wise NLI 是把规则比较升级成语义矛盾检测。

### 5.2 对每个字段构造 hypothesis

当前 caption 抽到字段后，我们把字段变成自然语言 hypothesis。

#### entity

字段值：

```text
Biden
```

hypothesis：

```text
The image event involves Biden.
```

#### location

字段值：

```text
Paris
```

hypothesis：

```text
The image event took place in Paris.
```

#### time

字段值：

```text
2024
```

hypothesis：

```text
The image event happened at 2024.
```

#### event_type

字段值：

```text
protest
```

hypothesis：

```text
The image event is about protest.
```

#### relation

字段值：

```text
arrest
```

hypothesis：

```text
The image event includes the relation or action: arrest.
```

### 5.3 premise 是什么

premise 不是当前 caption，而是：

```text
true_image_context
```

例如：

```text
true_image_context = "Police confronted demonstrators in London in 2019."
hypothesis = "The image event took place in Paris."
```

NLI 模型输出：

```json
{
  "entailment": 0.02,
  "neutral": 0.10,
  "contradiction": 0.88
}
```

于是：

```text
location -> contradiction -> location mismatch
```

### 5.4 阈值与决策

设置：

```text
contradiction_threshold = 0.60
entailment_threshold = 0.60
```

规则：

```text
if contradiction >= threshold:
    field = conflict
elif entailment >= threshold:
    field = consistent
else:
    field = neutral/uncertain
```

实际论文实验中，阈值应该在人工标注 dev set 上调参。当前默认阈值是初始值，不应说成最优。

### 5.5 输出

```json
{
  "field_nli": {
    "entity": {"label": "entailment", "scores": {...}},
    "location": {"label": "contradiction", "scores": {...}},
    "time": {"label": "neutral", "scores": {...}},
    "event_type": {"label": "entailment", "scores": {...}},
    "relation": {"label": "neutral", "scores": {...}}
  },
  "conflict_fields": ["location"],
  "mismatch_type": "location mismatch"
}
```

### 5.6 起到的效果

它直接回答：

```text
当前 caption 的哪个字段被 true context 反驳？
```

这才是解释模块的核心。

---

## 6. Step 5：Graph alignment，具体怎么做

### 6.1 为什么需要 graph alignment

Field-wise NLI 是字段级的，但某些误用发生在组合关系里。

例如：

```text
current_caption: Biden visited Ukraine.
true_context: Biden met officials in Washington.
```

单看实体：

```text
Biden 一致
```

但事件关系不同：

```text
visited Ukraine
vs
met officials in Washington
```

所以需要关系图。

### 6.2 图怎么构造

从 OpenIE/SRL-like triples 构造：

```json
{
  "nodes": ["Biden", "Ukraine"],
  "edges": [
    {"subject": "Biden", "predicate": "visited", "object": "Ukraine"}
  ]
}
```

true context：

```json
{
  "nodes": ["Biden", "officials", "Washington"],
  "edges": [
    {"subject": "Biden", "predicate": "met", "object": "officials"}
  ]
}
```

### 6.3 怎么对齐

对每条 current triple，找最相似的 true triple。

比较：

```text
subject similarity
predicate similarity
object similarity
```

如果：

```text
subject 高相似
predicate 中高相似
object 低相似
```

则可能是 relation/object grounding conflict。

输出：

```json
{
  "graph_alignment_score": 0.42,
  "graph_conflicts": ["relation"],
  "aligned_edges": [
    {
      "current": {"subject": "Biden", "predicate": "visited", "object": "Ukraine"},
      "true": {"subject": "Biden", "predicate": "met", "object": "officials"},
      "score": 0.41
    }
  ]
}
```

### 6.4 这一步和 Evidence-GNN 的关系

Evidence-GNN 类方法会把 claim 与 evidence 构造成图，并用 GNN 学习图间关系。我们现在不训练完整 GNN，而是迁移图对齐思想，做一个可执行、可解释、低成本的 graph matching baseline。后续如果有时间，可以把它替换成 GNN。

---

## 7. Step 6：最终归因决策

最终输入：

```text
VDT label
Evidence sufficiency
Field-wise NLI conflicts
Graph conflicts
```

决策逻辑：

```text
if evidence_sufficiency == insufficient:
    mismatch_type = uncertain / evidence insufficient
    conflict_fields = [evidence_insufficient]

elif field_nli has contradiction:
    conflict_fields = all contradiction fields
    mismatch_type = field with highest contradiction score

elif graph_alignment has conflict:
    conflict_fields = graph_conflicts
    mismatch_type = relation mismatch / context omission

else:
    mismatch_type = benign illustrative image or context omission
```

注意：

```text
这个归因结果不覆盖 VDT label。
```

最终解释：

```text
VDT detects OOC. The current caption claims the event happened in Paris, but the true image context supports London. Field-wise NLI marks the location hypothesis as contradiction, so the mismatch type is location mismatch.
```

---

## 8. 实验验证：所有主张怎么支撑

### 8.1 VDT 复现表

已完成。

### 8.2 COVE-lite coverage

验证 true context 构造是否可用。

```text
coverage = kept / total
```

### 8.3 Event extraction evaluation

人工标注 event tuple，比较：

```text
rule extractor
enhanced extractor
LLM JSON extractor（可选）
```

指标：

```text
Entity F1
Location F1
Time F1
Event Type F1
Relation F1
```

### 8.4 Attribution evaluation

人工标注：

```text
gold_mismatch_type
gold_conflict_fields
```

比较：

```text
rule sidecar
similarity-only
COVE-lite + rule
COVE-lite + field-wise NLI
COVE-lite + NLI + evidence relevance
COVE-lite + NLI + evidence relevance + graph alignment
```

指标：

```text
Mismatch Type Accuracy
Conflict Field Micro-F1
Conflict Field Macro-F1
Exact Match Rate
```

### 8.5 Evidence relevance ablation

比较：

```text
No relevance filter
Length-only filter
Overlap filter
NLI + overlap relevance
```

### 8.6 Hard negative evaluation

构造：

```text
same-topic different-location
same-person different-time
same-location different-event
same-event-type different-location
high similarity but OOC
```

验证：

```text
NLI/graph attribution 是否比 similarity-only 更能识别字段冲突。
```

---

## 9. 本地命令

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

如果模型下载失败，可先：

```powershell
-NoTransformers
```

但正式报告必须说明 fallback 不是主方法。

---

## 10. 答辩口径

> 我们不再把规则抽取包装成创新。规则模块仅作为 baseline。最终 VDT-COVE-Attr v2 采用 COVE-lite true context 构造图像真实语境，使用 enhanced event extraction 从 current caption 和 true context 中抽取事件字段，再使用 field-wise NLI 判断每个字段是否被真实上下文反驳，并结合 evidence relevance 避免证据不足时强行归因。Graph alignment 进一步补充 subject-predicate-object 关系冲突。最终解释模块通过人工 attribution gold set 的 Type Accuracy、Field Micro-F1、Macro-F1 和 Exact Match Rate 验证。
