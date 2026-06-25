# VDT-COVE-Attr 模块方法深挖版

> 这份文档专门回答答辩中最容易被追问的问题：  
> **每个模块到底是什么？用的什么方法？这个方法从哪里来？为什么选它？它对结果有什么影响？局限是什么？下一步怎么改？**  
>  
> 先说边界：我们已经做成一个完整系统闭环，但其中若干模块当前是 `lite / fallback / baseline` 版本。我们不把它们包装成完整论文模型。

---

## 0. 系统总览：每个模块在链路中的位置

```text
NewsCLIPpings sample
  ├─ id: 当前 caption 来源
  ├─ image_id: 当前图片来源
  └─ falsified: OOC 二分类标签

        ↓

[A] COVE-lite true context construction
  current_caption = VisualNews[id].caption
  true_image_context = VisualNews[image_id].caption

        ↓

[B] VDT baseline classification
  image + current_caption -> OOC / Non-OOC

        ↓

[C] Enhanced event extraction
  current_caption -> current_event
  true_context -> true_event

        ↓

[D] Evidence relevance / sufficiency
  判断 true_context 是否足够支持归因

        ↓

[E] Field-wise NLI
  对 entity/location/time/event_type/relation 分别判断
  entailment / neutral / contradiction

        ↓

[F] Graph alignment-lite
  对 OpenIE-like triples 做 subject/predicate/object 对齐

        ↓

[G] Attribution decision
  输出 mismatch_type + conflict_fields + explanation
```

一句话：

```text
VDT 判断“是不是 OOC”，VDT-COVE-Attr 判断“如果是/可能是 OOC，具体错在哪里，以及证据是否足够”。
```

---

## 1. 模块 A：COVE-lite true context construction

### 1.1 这个模块是什么

这个模块负责给每张图片找“原始语境”。

NewsCLIPpings 的 OOC 样本是通过重新配对 VisualNews 中的图片和文本构造出来的。因此一个样本里通常有：

```json
{
  "id": 714504,
  "image_id": 1670772,
  "falsified": true
}
```

含义是：

```text
id       -> 当前 caption 来自哪条 VisualNews 记录
image_id -> 当前图片来自哪条 VisualNews 记录
```

所以：

```text
current_caption     = VisualNews[id].caption
true_image_context  = VisualNews[image_id].caption
```

这一步得到的 `true_image_context` 是后续所有解释的基础。

### 1.2 它用的是什么方法

当前用的是 **数据关系回查**，不是训练模型。

代码入口：

```text
scripts/context/build_cove_lite_context_pairs.py
```

具体操作：

```text
1. 读取 NewsCLIPpings 的 train/val/test JSON。
2. 读取 VisualNews 的 origin/data.json。
3. 建立 id -> caption/context 的字典。
4. 对每条 NewsCLIPpings 样本：
   - 用 id 找 current_caption。
   - 用 image_id 找 true_image_context。
5. 输出 cove_lite_context_pairs.jsonl。
```

当前优先数据源：

```text
E:\OOC_Datasets\VisualNews\origin\data.json
```

`articles_metadata/*.p` 只作为 fallback，因为我们检查发现其中很多记录只有 `article_path / image_id / id`，没有 caption 文本。

### 1.3 这是已有方法还是我们的结合

思想来自 COVE，但实现是我们的轻量适配。

COVE 的核心是：

```text
先获得图片真实上下文，再用真实上下文判断当前 caption 是否可信。
```

COVE 论文明确把 OOC debunking 拆成两个目标：提供图片 true context，以及判断 caption veracity；它提出先预测图像 context，再判断 caption veracity 的流程。我们没有完整复现它的 context prediction 模型，而是利用 NewsCLIPpings / VisualNews 的已知 id 对齐关系，直接取原始 caption 作为 true context。参考：[COVE, NAACL 2025 / ACL Anthology](https://aclanthology.org/2025.naacl-long.102/)。

所以正确表述是：

```text
我们迁移 COVE 的 context-first 思路，做了 COVE-lite 数据构造。
```

不能说：

```text
我们完整实现了 COVE。
```

### 1.4 为什么选这个方法

因为它最稳、最可复现。

如果我们用检索或大模型生成 true context，会有三个问题：

```text
1. 结果不稳定，答辩现场和队友电脑上可能不一致。
2. 成本高，需要联网、模型或 API。
3. 难以证明生成的 true context 一定对应这张图片。
```

而 NewsCLIPpings 本身来自 VisualNews，`image_id` 已经指向图片原始记录，所以直接回查是最干净的数据构造方式。

### 1.5 对结果有什么影响

它决定解释模块有没有可靠证据。

如果没有 `true_image_context`，后面 NLI 和 graph alignment 都只能对空证据工作，结果必然不可信。

当前 smoke test 显示：

```text
MaxRecords = 200
kept = 200
missing_text = 0
missing_true_context = 0
```

说明数据链路已经打通。

### 1.6 老师可能追问

**问：为什么 true context 可以代表图片真实语境？**

答：因为 NewsCLIPpings 样本来自 VisualNews 的重新配对。`image_id` 对应图片在 VisualNews 中的原始记录，原始 caption 是这张图片在数据集中的真实新闻上下文。因此它可以作为 COVE-lite 的 evidence。

**问：caption 会不会太短？**

答：会。这也是我们加入 evidence sufficiency 的原因。如果 true context 太短或字段太少，系统输出 evidence insufficient，不强行解释。

### 1.7 下一步怎么改

```text
短期：把 VisualNews caption + topic + source 拼成更丰富 true context。
中期：读取 article_path 的文章标题或首段，补充证据。
长期：接入检索式 COVE，给真实图片找多条外部 evidence，并做 evidence reranking。
```

---

## 2. 模块 B：VDT baseline classification

### 2.1 这个模块是什么

VDT 是主分类 baseline。

它回答：

```text
image + current_caption -> OOC / Non-OOC
```

在我们系统里，VDT 的角色很明确：

```text
VDT 决定是否 OOC。
VDT-COVE-Attr 只解释错在哪里。
```

### 2.2 它用的是什么方法

我们不是重新发明 VDT，而是复现已有论文/代码里的 VDT strict BLIP-2/GaussianBlur 设置。

已完成结果：

| Setting | F1 | Acc | AUC | 状态 |
|---|---:|---:|---:|---|
| bbc,guardian bs128 ep20 | 0.7353 | 0.7383 | 0.7398 | completed |
| usa_today,washington_post bs64 ep20 | 0.8032 | 0.8032 | 0.8028 | completed |
| usa_today,washington_post bs128 ep20 | - | - | - | CUDA OOM |

### 2.3 为什么选 VDT

因为项目最开始目标就是在 VDT 论文基础上做复现和扩展，而且 VDT 已经解决 OOC 主分类问题。

直接换掉 VDT 会带来两个风险：

```text
1. 复现主线不稳定。
2. 解释模块和分类模块混在一起，无法判断改进来自哪里。
```

所以我们把 VDT frozen，新增 sidecar。

### 2.4 对结果有什么影响

好处：

```text
1. 主分类指标可直接沿用 VDT 复现结果。
2. 解释模块失败不会污染分类准确率。
3. 可以单独评估 attribution 模块。
```

风险：

```text
1. 如果 VDT 分类错了，sidecar 默认不修正它。
2. 当前系统不是端到端联合优化。
```

### 2.5 老师可能追问

**问：你们是否牺牲准确率换解释？**

答：默认不牺牲。因为 sidecar 不覆盖 VDT label。分类结果仍由 VDT 输出，解释结果单独输出。

**问：为什么不把解释信号融合回分类？**

答：可以做，但必须实验验证不降 Acc/F1。当前为了满足“分类准确率不降低”的要求，先采用 baseline-preserving sidecar。

### 2.6 下一步怎么改

```text
1. 做 warning-only：VDT label 不变，但 sidecar 标出冲突 warning。
2. 做 calibrated fusion：只有高置信 NLI contradiction 才调整分类分数。
3. 比较 VDT vs VDT+sidecar fusion 的 Acc/F1/AUC。
```

---

## 3. 模块 C：Enhanced event extraction

这个模块是最容易被问细节的模块。它的任务是把自然语言拆成可以比较的事件字段。

输入：

```text
current_caption
true_image_context
```

输出：

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

代码入口：

```text
src/e3vdt/event/enhanced_extractor.py
scripts/event/extract_event_tuples_v2.py
```

---

### 3.1 子模块 C1：Rule fallback

#### 3.1.1 它是什么

Rule fallback 是最基础的事件字段抽取器。

它用：

```text
正则表达式
关键词表
小型地点词表
大写短语规则
```

抽取五类字段。

#### 3.1.2 它做了什么

例如：

```text
"A protest erupted in Paris on Monday."
```

规则会抽：

```json
{
  "locations": ["paris"],
  "times": ["Monday"],
  "event_types": ["protest"],
  "relations": ["gather"]
}
```

#### 3.1.3 为什么需要它

因为它：

```text
1. 无依赖，任何电脑都能跑。
2. 可审计，老师问到可以直接解释。
3. 可以作为 baseline。
4. 当 spaCy/NLI 模型不可用时兜底。
```

#### 3.1.4 它对结果的影响

正面影响：

```text
保证系统不会因为模型缺失而完全不能运行。
能快速生成 weak attribution baseline。
```

负面影响：

```text
覆盖率和准确率有限。
复杂实体、隐式地点、复杂关系容易漏。
```

#### 3.1.5 改进

```text
1. 扩充事件关键词。
2. 扩充地点 alias。
3. 用人工标注集评估每个字段 F1。
4. 逐步让 rule fallback 只做兜底，而不是主抽取器。
```

---

### 3.2 子模块 C2：spaCy NER

#### 3.2.1 spaCy NER 是什么

spaCy 是一个开源 NLP 工具库。NER 是 Named Entity Recognition，命名实体识别。

NER 的任务是从文本中识别“有名称的现实对象”，例如：

```text
PERSON：Biden, Obama
ORG：United Nations, Community Health Systems
GPE：France, Washington, London
DATE：2024, Monday
```

spaCy 官方文档把 `EntityRecognizer` 描述为一个 transition-based NER 组件，用来识别非重叠的实体 span。spaCy 101 也说明，named entity 是带名称的现实对象，模型会预测文本中的实体类型。参考：[spaCy EntityRecognizer API](https://spacy.io/api/entityrecognizer)，[spaCy 101 Named Entities](https://spacy.io/usage/spacy-101)。

#### 3.2.2 它用什么方法做

在当前项目里，我们调用的是 spaCy 预训练英文模型：

```text
en_core_web_sm
```

代码：

```python
doc = nlp(text)
for ent in doc.ents:
    ...
```

映射关系：

```text
PERSON / ORG / NORP -> entities
GPE / LOC / FAC -> locations
DATE / TIME -> times
```

也就是说，spaCy 负责从新闻 caption 中自动识别人名、组织、国家、地点和时间表达。

#### 3.2.3 这是已有方法还是我们的结合

spaCy NER 是已有成熟 NLP 工具，不是我们的新方法。

我们的结合点是：

```text
把通用 NER 输出映射到 OOC attribution 需要的 event fields。
```

例如 spaCy 输出：

```text
Biden -> PERSON
Washington -> GPE
2024 -> DATE
```

我们转成：

```json
{
  "entities": ["biden"],
  "locations": ["washington"],
  "times": ["2024"]
}
```

#### 3.2.4 为什么选 spaCy

原因很实际：

```text
1. 易安装、易复现。
2. 推理速度快，适合本地课程项目。
3. 输出标签直接覆盖 person/org/location/date。
4. 不需要我们重新标注和训练 NER。
5. 可以作为 rule extractor 的增强层。
```

我们没有选择更重的事件抽取模型，是因为课程项目时间有限，且我们还需要系统演示、VDT 复现、NLI 归因和人工评估。

#### 3.2.5 它对结果有什么影响

spaCy NER 主要提升三个字段：

```text
entity
location
time
```

如果只靠规则，大写短语和小词表可能漏掉很多实体。spaCy 可以提升字段覆盖率。

你的 smoke test 中：

```text
current_entities = 200 / 200
true_entities = 200 / 200
current_locations = 121 / 200
true_locations = 130 / 200
```

这说明 enhanced extractor 已经能稳定地产生实体和一部分地点字段。

#### 3.2.6 它的局限

```text
1. en_core_web_sm 是小模型，复杂新闻实体可能识别错。
2. 中文或多语言 caption 支持有限。
3. NER 只识别实体，不理解事件逻辑。
4. 它不知道 OOC 任务里什么字段最关键。
5. 它可能把地点识别成组织，或者把组织识别成地点。
```

所以答辩时要说：

```text
spaCy NER 是 enhanced extractor 的一个预训练信息抽取层，不是最终事件理解模型。
```

#### 3.2.7 老师可能追问

**问：spaCy NER 是你们训练的吗？**

答：不是。我们使用已有预训练 NER 模型，把它映射到我们的 event tuple schema。

**问：为什么不用 LLM 抽取？**

答：LLM 可以作为后续增强，但课程系统需要可复现和可离线运行。spaCy 更稳定、成本低、容易让队友复现。

#### 3.2.8 改进

```text
1. 换更强 spaCy transformer 模型。
2. 使用 Stanza / Flair / HanLP 做多工具对比。
3. 对 100-300 条样本人工标注 entity/location/time，计算字段 F1。
4. 引入 LLM JSON extractor 作为候选抽取器，但必须缓存输出并评估。
```

---

### 3.3 子模块 C3：字段归一化

#### 3.3.1 它是什么

字段归一化是把不同表面写法统一成同一个标准形式。

代码：

```text
src/e3vdt/event/normalize.py
```

#### 3.3.2 它做了什么

地点归一化：

```text
U.S. / US / USA / America -> united states
UK / Britain / England -> united kingdom
Washington D.C. -> washington
```

事件类型归一化：

```text
war / attack / strike -> war/conflict
rally / demonstration / march -> protest
vote / election -> politics
trial / arrest -> court/crime
```

#### 3.3.3 为什么需要

如果不归一化，系统会把同义表达误判成冲突。

例如：

```text
current: US
true: United States
```

字符串不同，但语义相同。归一化后都变成：

```text
united states
```

#### 3.3.4 对结果的影响

它主要减少 false positive。

```text
没有归一化：可能误判 location mismatch。
有归一化：同义地点更可能被视为一致。
```

#### 3.3.5 局限与改进

当前 alias 表很小。

改进：

```text
1. 扩充国家、城市、组织地点 alias。
2. 引入 GeoNames / Wikidata 做标准化。
3. 时间字段接 dateparser / SUTime / HeidelTime。
4. 事件类型设计更清晰 taxonomy。
```

---

### 3.4 子模块 C4：OpenIE-like triples

#### 3.4.1 OpenIE 是什么

OpenIE 是 Open Information Extraction，开放信息抽取。

它的目标是从非结构化文本中抽取关系元组，例如：

```text
Police arrested demonstrators.
```

抽成：

```text
(Police, arrested, demonstrators)
```

OpenIE 早期代表工作 Banko et al. 2007 提出从 Web-scale 文本中开放式抽取关系，不预先限定关系类型。参考：[Open Information Extraction from the Web, IJCAI 2007](https://www.ijcai.org/Proceedings/07/Papers/429.pdf)。

#### 3.4.2 我们当前做了什么

我们没有接完整 OpenIE 系统，而是实现了轻量 fallback：

```text
src/e3vdt/event/enhanced_extractor.py::extract_openie_like_triples
```

当前支持两类模板：

```text
1. 主体 + 动词 + 宾语
2. 主体 + in/at/near + 地点
```

例子：

```text
Police arrested demonstrators near the court.
```

输出：

```json
{
  "subject": "Police",
  "predicate": "arrested",
  "object": "demonstrators near the court"
}
```

#### 3.4.3 这是已有方法还是我们自己的实现

思想来自 OpenIE/SRL，但当前代码是我们自己的轻量正则实现。

准确说法：

```text
我们实现了 OpenIE/SRL-like fallback，用于提供 relation-level graph alignment 输入。
```

不能说：

```text
我们实现了完整 OpenIE 或 SRL。
```

#### 3.4.4 为什么需要它

字段级信息不一定能表达关系错配。

例如：

```text
Biden visited Ukraine.
Biden met officials in Washington.
```

二者都可能有：

```text
entity = Biden
```

但关系不同：

```text
visited Ukraine
met officials
```

所以需要三元组。

#### 3.4.5 对结果的影响

它让系统能够输出：

```text
relation mismatch
graph_conflicts = ["relation"]
aligned_edges = [...]
```

也让后续 Evidence-GNN 替换有接口。

#### 3.4.6 局限与改进

局限：

```text
1. 正则覆盖很少。
2. 不能处理长句、从句、被动句。
3. 不能识别复杂谓词-论元结构。
4. 不能处理跨句关系。
```

改进：

```text
1. 接 Stanford OpenIE / OpenIE6。
2. 接 SRL 模型抽 predicate-argument。
3. 接 AMR parser，构造语义图。
4. 用人工 relation gold set 验证 relation F1。
```

---

## 4. 模块 D：Evidence relevance / sufficiency

### 4.1 这个模块是什么

这个模块判断 true_image_context 是否足够支持后续归因。

如果证据太短、太空、字段太少，系统不应该强行解释。

### 4.2 它用什么方法

当前是轻量启发式打分。

代码：

```text
src/e3vdt/attribution/evidence_relevance_v2.py
```

计算：

```text
relevance =
  0.35 * text_similarity
+ 0.35 * field_overlap_mean
+ 0.15 * context_len_score
+ 0.15 * information_score
```

其中：

```text
text_similarity：current_caption 和 true_context 的文本相似度。
field_overlap_mean：entity/location/time/event_type/relation 的平均重合度。
context_len_score：true_context 长度是否足够。
information_score：true_event 里非空字段数量。
```

### 4.3 方法依据是什么

思想来自 evidence-based fact-checking 里的 relevant evidence detection。

RED-DOT 指出，不能默认所有外部证据都相关；它提出 Relevant Evidence Detection，用于判断证据是否支持或反驳 claim。参考：[RED-DOT arXiv](https://arxiv.org/abs/2311.09939)，[RED-DOT GitHub](https://github.com/stevejpapad/relevant-evidence-detection)。

我们没有复现 RED-DOT 模型，而是迁移它的核心思想：

```text
先判断证据是否足够，再做归因。
```

### 4.4 为什么选这个方法

因为我们当前的 true context 有时只是 caption，信息量可能有限。

如果没有 sufficiency gate，系统可能会：

```text
true_context = "A file photo"
current_caption = "Protesters gathered in Paris"
输出 location mismatch
```

这明显不可靠。

### 4.5 对结果有什么影响

它降低强行解释的风险。

最终决策中：

```python
if evidence_sufficiency != "sufficient":
    mismatch_type = "uncertain / evidence insufficient"
    conflict_fields = ["evidence_insufficient"]
```

这意味着：

```text
证据不足时，系统优先承认不能判断。
```

### 4.6 局限

当前 relevance 还是启发式：

```text
1. 权重 0.35/0.35/0.15/0.15 是初始设计，不是调参最优。
2. text_similarity 是表面相似度，不是真正语义相关性。
3. field_overlap 依赖事件抽取质量。
4. 没有训练 relevance classifier。
```

### 4.7 改进

```text
1. 用人工 gold set 调 sufficient_threshold。
2. 引入 NLI entailment/contradiction 分数参与 relevance。
3. 训练二分类 evidence relevance classifier。
4. 做 ablation：无 gate / length-only / overlap / NLI+overlap。
```

---

## 5. 模块 E：Field-wise NLI

### 5.1 这个模块是什么

Field-wise NLI 是当前 v2 归因主方法。

它不直接比较整句，而是逐字段问：

```text
true_image_context 是否支持 current_caption 中的 entity 声明？
true_image_context 是否支持 current_caption 中的 location 声明？
true_image_context 是否支持 current_caption 中的 time 声明？
...
```

### 5.2 NLI 是什么

NLI 是 Natural Language Inference，自然语言推理。

输入：

```text
premise
hypothesis
```

输出：

```text
entailment：premise 支持 hypothesis
neutral：premise 不能判断 hypothesis
contradiction：premise 反驳 hypothesis
```

当前模型：

```text
facebook/bart-large-mnli
```

Hugging Face model card 说明，这类预训练 NLI 模型可用于 zero-shot classification：把待判断文本作为 premise，把候选标签构造成 hypothesis，再看 entailment/contradiction。参考：[facebook/bart-large-mnli model card](https://huggingface.co/facebook/bart-large-mnli)。

我们借用的是同一思想，但不是做普通分类，而是做字段级语境矛盾判断。

### 5.3 它具体怎么做

代码：

```text
src/e3vdt/attribution/field_nli_v2.py
```

对每个字段构造 hypothesis：

| 字段 | hypothesis 模板 |
|---|---|
| entity | `The image event involves {values}.` |
| location | `The image event took place in {values}.` |
| time | `The image event happened at {values}.` |
| event_type | `The image event is about {values}.` |
| relation | `The image event includes the relation or action: {values}.` |

然后：

```text
premise = true_image_context
hypothesis = 字段模板句
```

例如：

```text
true_context:
People gathered in London during a climate demonstration in 2020.

hypothesis:
The image event took place in Paris.
```

如果模型输出 contradiction 高于阈值，则：

```text
location -> conflict
mismatch_type -> location mismatch
```

### 5.4 为什么选择 NLI

因为我们要判断的是“语义矛盾”，不是简单相似度。

整句相似度有 shortcut 问题。MUSE 工作指出，OOC 检测中简单相似度 baseline 可能很强，但这种成功不一定代表模型真正理解事实一致性和逻辑矛盾。参考：[MUSE / Similarity over Factuality](https://arxiv.org/abs/2407.13488)。

因此我们需要从：

```text
相似不相似
```

升级到：

```text
是否被真实上下文支持 / 反驳 / 无法判断
```

NLI 正适合这个接口。

### 5.5 对结果有什么影响

NLI 让输出从：

```text
字符串相似度低，所以可能错。
```

变成：

```text
true context 反驳了 current caption 的 location hypothesis，所以是 location mismatch。
```

输出更像事实核验解释。

### 5.6 当前正式 NLI 跑什么

你现在跑：

```powershell
-NliModel facebook/bart-large-mnli
-NliDevice 0
```

是在做：

```text
对 500 条样本的每个字段运行 BART-MNLI 推理。
```

不是训练。

跑完后必须看：

```text
used_transformers = true
nli_backend_counts = facebook/bart-large-mnli
```

如果是：

```text
lexical_fallback
```

说明没跑真实 NLI，不能作为正式结果。

### 5.7 局限

```text
1. BART-MNLI 是通用 NLI，不是 OOC 专用模型。
2. 字段 hypothesis 模板可能影响判断。
3. 多个实体拼在一个 hypothesis 中可能干扰模型。
4. 短 caption 可能没有足够 premise。
5. NLI 的 contradiction 阈值需要用 dev/gold set 调。
```

### 5.8 改进

```text
1. 一个字段多个值时逐个值跑 NLI，再聚合。
2. 试不同 hypothesis 模板。
3. 使用事实核验领域 NLI 模型。
4. 用人工 gold set 调阈值。
5. 把 NLI 分数和 graph/evidence 特征输入轻量分类器。
```

---

## 6. 模块 F：Graph alignment-lite

### 6.1 这个模块是什么

Graph alignment-lite 用来补充关系级冲突。

Field-wise NLI 主要看字段集合，但一些错误发生在关系组合上：

```text
Biden visited Ukraine.
Biden met officials in Washington.
```

实体可能一致，但关系不同。

### 6.2 方法来源是什么

它借鉴的是 graph-based multimodal misinformation detection 的思想。

Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs 把 caption claim 和 textual evidence 构造成 claim graph / evidence graph，再用 GNN 编码和比较。参考：[Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs](https://arxiv.org/abs/2505.18221)。

我们当前没有训练 GNN，只做轻量 graph matching。

### 6.3 当前怎么实现

代码：

```text
src/e3vdt/attribution/graph_alignment_v2.py
```

输入：

```text
current_event.relations_structured
true_event.relations_structured
```

每个 triple：

```json
{
  "subject": "Biden",
  "predicate": "visited",
  "object": "Ukraine"
}
```

对齐方式：

```text
1. 对每条 current triple，找最相似的 true triple。
2. 分别计算 subject/predicate/object 的字符串相似度。
3. 三者平均作为 edge alignment score。
4. 如果 subject 和 predicate 相似，但 object 差异大，则标记 relation conflict。
```

冲突规则：

```text
subject_sim >= 0.65
predicate_sim >= 0.45
object_sim < 0.35
```

则：

```text
graph_conflicts = ["relation"]
```

### 6.4 为什么需要它

因为 NLI 字段判断可能漏掉关系层面的错配。

Graph alignment 提供：

```text
1. relation-level evidence
2. aligned_edges 可解释输出
3. 后续替换为 Evidence-GNN 的接口
```

### 6.5 对结果有什么影响

如果 graph alignment 发现冲突，最终 decision 会把它加入：

```text
v2_conflict_fields
```

并可能输出：

```text
relation mismatch
```

### 6.6 局限

```text
1. triples 来源是 OpenIE-like fallback，不完整。
2. similarity 是字符串相似度，不是语义 embedding。
3. 没有 GNN，没有 attention，没有训练。
4. relation conflict 规则是启发式。
```

### 6.7 改进

```text
1. 用真正 OpenIE/SRL/AMR 构造图。
2. 节点和边使用 sentence embedding。
3. 构造 claim graph / evidence graph。
4. 训练 GNN 或 graph attention classifier。
5. 做 relation mismatch 子集评估。
```

---

## 7. 模块 G：最终归因决策

### 7.1 它是什么

最终归因决策把三个模块合并：

```text
Evidence relevance
Field-wise NLI
Graph alignment-lite
```

输出：

```text
v2_mismatch_type
v2_conflict_fields
v2_attribution_reason
```

### 7.2 当前怎么做

代码：

```text
scripts/attribution/run_field_nli_attribution_v2.py::final_decision
```

优先级：

```text
1. evidence insufficient -> uncertain / evidence insufficient
2. field NLI contradiction -> 对应 mismatch type
3. graph conflict -> relation mismatch / context omission
4. 无冲突 -> benign illustrative image
```

### 7.3 为什么这样排序

证据不足优先级最高，因为没有足够 evidence 时，任何解释都不可靠。

然后是 NLI，因为它直接判断字段是否被 true context 反驳。

最后是 graph，因为当前 graph 是 lite 版本，作为补充关系证据。

### 7.4 对结果有什么影响

这个模块决定用户最终看到的解释。

例如：

```json
{
  "v2_mismatch_type": "location mismatch",
  "v2_conflict_fields": ["location", "time"],
  "v2_attribution_reason": "field_or_graph_contradiction"
}
```

### 7.5 局限

```text
1. 当前是规则式决策。
2. 没有学习不同模块权重。
3. 对多字段冲突的主类型选择还比较粗。
4. benign illustrative image 只是“未发现冲突”，不是绝对真实。
```

### 7.6 改进

```text
1. 使用人工 gold set 训练 attribution head。
2. 输入 NLI scores、evidence relevance、graph scores。
3. 输出 multi-label conflict_fields。
4. 做 confidence calibration。
```

---

## 8. 模块 H：人工归因评估

### 8.1 为什么必须人工评估

NewsCLIPpings 只有二分类标签，没有：

```text
mismatch_type
conflict_fields
```

所以解释模块不能只靠自己输出证明自己正确。

必须人工标注：

```text
gold_mismatch_type
gold_conflict_fields
```

### 8.2 当前怎么做

候选生成：

```text
scripts/eval/build_attribution_eval_sample.py
```

输出：

```text
examples/attribution_eval_candidates.jsonl
```

人工填完后保存为：

```text
examples/attribution_eval_set.jsonl
```

评估脚本：

```text
scripts/eval/evaluate_attribution_v2.py
```

指标：

```text
Mismatch Type Accuracy
Conflict Field Micro Precision / Recall / F1
Conflict Field Macro-F1
Exact Match Rate
```

### 8.3 为什么这些指标合适

```text
Type Accuracy：主错配类型是否判断对。
Field Micro-F1：所有字段整体预测是否对。
Field Macro-F1：避免某个高频字段掩盖低频字段。
Exact Match：整条样本所有冲突字段是否完全一致。
```

### 8.4 评估什么方法

当前至少比较：

```text
weak_rule_sidecar
v2_field_nli_evidence_graph
```

后续应加入：

```text
majority baseline
random baseline
similarity-only / MUSE-lite baseline
no evidence relevance ablation
no graph alignment ablation
```

### 8.5 方法来源

MUSE 提醒我们，简单相似度 baseline 在 OOC 上可能很强，但可能只是利用表面 shortcut，不是真正 factual reasoning。因此我们必须把 similarity-only 纳入 baseline，而不是只和 random 比。参考：[MUSE / WACV 2025](https://arxiv.org/abs/2407.13488)。

### 8.6 改进

```text
1. 标注 100-300 条。
2. 双人标注，计算一致性。
3. hard negative 分层评估。
4. 使用 gold set 调 NLI 阈值。
5. 报告每个 mismatch type 的混淆矩阵。
```

---

## 9. 模块 I：Gradio 展示系统

### 9.1 它是什么

展示系统是答辩现场用的可视化界面。

入口：

```text
demo/app.py
```

运行：

```powershell
python demo/app.py
```

### 9.2 它展示什么

主要展示：

```text
1. VDT-COVE-Attr 主系统
2. 分类不降验证
3. 复现实验指标
4. 实验看板
```

### 9.3 为什么需要它

因为答辩不能只讲脚本和 JSON。展示系统能直观看到：

```text
current_caption
true_image_context
field-wise attribution
evidence sufficiency
conflict_fields
final explanation
```

### 9.4 对结果有什么影响

展示系统不产生最终实验指标。它的作用是：

```text
1. 证明系统闭环已经打通。
2. 给老师展示输入输出。
3. 给队友调试样例。
```

不能说：

```text
demo case 指标就是论文结果。
```

### 9.5 改进

```text
1. 显示 NLI 三分类概率条。
2. 显示 graph aligned_edges。
3. 支持上传图片并调用真实 VDT checkpoint。
4. 支持导出单样本报告。
```

---

## 10. 各模块一句话答辩口径

| 模块 | 一句话口径 |
|---|---|
| COVE-lite | 利用 NewsCLIPpings 与 VisualNews 的 id 对齐关系，为每张图片构造 true image context。 |
| VDT | 保留 VDT 作为主分类器，解释模块不覆盖分类 label，保证不降准确率。 |
| Rule fallback | 无依赖的弱抽取 baseline，用于兜底和对照，不作为最终高标准方法。 |
| spaCy NER | 使用已有预训练 NER 工具抽取人物、组织、地点、时间，并映射到 event tuple。 |
| normalization | 把 US/USA/United States 等同义表达统一，减少假冲突。 |
| OpenIE-like triples | 用轻量关系三元组抽取为 relation-level attribution 和 graph alignment 提供输入。 |
| Evidence relevance | 判断 true context 是否足够支持解释，证据不足时拒绝强行归因。 |
| Field-wise NLI | 把字段声明构造成 hypothesis，用 BART-MNLI 判断是否被 true context 反驳。 |
| Graph alignment-lite | 对 subject-predicate-object 三元组做轻量对齐，补充关系级冲突。 |
| Human gold eval | 用人工标注的 mismatch_type 和 conflict_fields 验证解释模块是否可靠。 |

---

## 11. 最终不能说错的边界

必须避免这些说法：

```text
我们实现了完整 COVE。
我们实现了完整 OpenIE/SRL。
我们实现了 Evidence-GNN。
NLI 已经证明一定优于规则。
demo case 指标就是最终实验指标。
我们超过了 VDT 主分类。
```

应该说：

```text
我们实现了 COVE-lite、OpenIE-like fallback、graph alignment-lite。
我们用 BART-MNLI 做字段级语义矛盾判断。
当前系统闭环已经完成，归因有效性需要人工 gold set 和 ablation 支撑。
```

---

## 12. 参考资料

1. spaCy EntityRecognizer API: <https://spacy.io/api/entityrecognizer>  
2. spaCy 101 Named Entities: <https://spacy.io/usage/spacy-101>  
3. Hugging Face `facebook/bart-large-mnli` model card: <https://huggingface.co/facebook/bart-large-mnli>  
4. Banko et al., Open Information Extraction from the Web: <https://www.ijcai.org/Proceedings/07/Papers/429.pdf>  
5. COVE: COntext and VEracity prediction for out-of-context images: <https://aclanthology.org/2025.naacl-long.102/>  
6. RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection: <https://arxiv.org/abs/2311.09939>  
7. MUSE / Similarity over Factuality: <https://arxiv.org/abs/2407.13488>  
8. Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs: <https://arxiv.org/abs/2505.18221>  
9. Interpretable Detection of Out-of-Context Misinformation with Neural-Symbolic-Enhanced Large Multimodal Model: <https://arxiv.org/abs/2304.07633>

