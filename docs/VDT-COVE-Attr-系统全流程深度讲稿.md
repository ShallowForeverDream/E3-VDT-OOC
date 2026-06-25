# VDT-COVE-Attr 系统全流程深度讲稿

> 适用场景：课程答辩、项目汇报、队友分工理解、老师追问准备。  
> 核心原则：只讲已经实现或已经明确设计的内容；没有完成的部分明确说“当前是 lite / fallback / baseline”，不包装成完整 SOTA。  
> 当前项目路径：`D:\MY_PROJECT\OOC\E3-VDT-OOC`  
> 当前主线：**VDT 主分类 + COVE-lite true context + enhanced event extraction + field-wise NLI + evidence relevance + graph alignment-lite + 人工 attribution 评估协议**。

---

## 0. 先用一句话讲清楚整个系统

我们的系统不是重新训练一个新的 OOC 分类器，而是在已经复现的 VDT baseline 上增加一个可解释归因 sidecar。

VDT 负责回答：

```text
这组 image + current caption 是否 out-of-context？
```

我们新增的 VDT-COVE-Attr 负责回答：

```text
如果它是 OOC，错在哪里？
是人物错、地点错、时间错、事件类型错、关系错，还是证据不足不能判断？
```

完整流程是：

```text
NewsCLIPpings 样本
  -> VDT baseline 主分类
  -> COVE-lite 根据 image_id 找 VisualNews true_image_context
  -> current_caption 和 true_image_context 分别抽取事件字段
  -> evidence relevance 判断 true context 是否足够解释
  -> field-wise NLI 判断字段是否被真实上下文反驳
  -> graph alignment-lite 补充关系三元组冲突
  -> 输出 mismatch_type、conflict_fields、解释文本、JSON 结果
```

当前实现状态必须这样讲：

| 模块 | 当前状态 | 不能夸大的地方 |
|---|---|---|
| VDT 主分类 | 已完成两组 strict baseline 复现 | 不说全面超过论文，只说完成核心设置复现 |
| COVE-lite true context | 已实现，优先读取 `VisualNews/origin/data.json` | 不说完整 COVE 检索/生成模型 |
| Event extraction | 已实现 enhanced extractor | 当前是 rule + spaCy + OpenIE-like fallback，不是完整事件抽取 SOTA |
| Field-wise NLI | 已实现 HuggingFace NLI 推理接口 | 跑 `facebook/bart-large-mnli` 是推理，不是训练 |
| Evidence relevance | 已实现启发式 sufficiency/relevance gate | 当前不是训练好的 RED-DOT 模型 |
| Graph alignment | 已实现 lightweight graph matching | 当前不是 Evidence-GNN 训练模型 |
| 人工归因评估 | 已有 candidates/gold/eval 脚本 | 需要人工 gold set 后才能给最终解释指标 |
| Gradio 展示系统 | 已完成 | 演示集指标不是最终论文实验指标 |

---

## 1. 项目问题定义：我们到底解决什么问题

### 1.1 任务背景

Out-of-Context Misinformation 指的是：图片本身可能是真的，文字本身也可能是真的，但图片和文字被错误组合后，形成了错误语境。

例如：

```text
图片真实语境：2019 年伦敦气候抗议现场。
当前配文：2024 年巴黎爆发抗议。
```

图片可能确实是抗议，文字也可能确实是抗议，但二者不是同一事件。这样的错误组合会造成“旧图新用”“地点偷换”“主体偷换”“事件挪用”。

### 1.2 VDT 已经解决什么

VDT 解决的是二分类：

```text
image + caption -> OOC / Non-OOC
```

它擅长在跨域场景下判断图文是否匹配，但它的输出主要是：

```text
OOC 或 Non-OOC
score / probability
```

它没有直接回答：

```text
为什么 OOC？
错在哪个字段？
是时间错还是地点错？
是否因为证据不足不能解释？
```

### 1.3 我们新增的问题

我们的问题是：

```text
在不破坏 VDT 主分类结果的前提下，为 OOC 检测补充可解释错配归因。
```

形式化地说：

```text
输入：image_id, current_caption, VDT_label, VDT_score
外部上下文：VisualNews true_image_context
输出：mismatch_type, conflict_fields, explanation, evidence status
```

其中：

```text
mismatch_type ∈ {
  entity mismatch,
  location mismatch,
  temporal mismatch,
  event-type mismatch,
  relation mismatch,
  uncertain / evidence insufficient,
  benign illustrative image
}
```

```text
conflict_fields ⊆ {
  entity,
  location,
  time,
  event_type,
  relation,
  evidence_insufficient
}
```

### 1.4 第一轮追问：为什么不直接让 VDT 输出解释？

答：VDT 原模型不是为字段级归因训练的。它的训练目标是 OOC 二分类，不包含 `mismatch_type` 或 `conflict_fields` 标签。NewsCLIPpings 原始数据也只有图文是否匹配，并没有“错配类型”标注。

如果强行让 VDT 二分类输出解释，会有两个问题：

```text
1. 没有监督标签支撑，解释不可验证。
2. 修改 VDT 分类头可能降低已经复现的主分类指标。
```

所以我们采用 sidecar：

```text
VDT 继续负责主分类；
归因模块只负责解释；
归因模块默认不覆盖 VDT label。
```

### 1.5 第二轮追问：这样是不是没有创新，只是加规则？

答：不能把“加规则”包装成创新。我们当前路线的创新点不在规则本身，而在于把 VDT 的二分类检测扩展成一个**可证据约束、可评估、可替换模块的错配归因框架**。

具体包括：

```text
1. COVE-lite：用 VisualNews 原始上下文构造 true image context。
2. Field-wise NLI：把 current caption 的字段声明转成 hypothesis，用 true context 做 premise，判断字段级矛盾。
3. Evidence relevance：证据不足时不强行解释。
4. Graph alignment-lite：把关系三元组作为关系级补充证据。
5. 人工 attribution gold set：用 Type Acc / Field-F1 验证解释是否可靠。
```

当前规则只作为 baseline 和 fallback，不作为最终高标准方法本身。

### 1.6 进一步改进

后续可以从三个层次升级：

```text
短期：扩大人工 gold set，验证 NLI 相比 rule baseline 是否有效。
中期：替换 OpenIE-like fallback 为 OpenIE/SRL/AMR parser。
长期：训练 Evidence-GNN 或 attribution head，让字段冲突判断从规则/NLI组合变成可学习模型。
```

---

## 2. 数据层：NewsCLIPpings 与 VisualNews 如何对应

### 2.1 原始数据是什么

我们使用的复现数据是 NewsCLIPpings / VisualNews。

一个 NewsCLIPpings 样本大致包含：

```json
{
  "id": 714504,
  "image_id": 1670772,
  "similarity_score": 0.73,
  "source_dataset": 2,
  "falsified": true
}
```

含义是：

```text
id：当前 caption/text 的 VisualNews id。
image_id：当前配对图片的 VisualNews id。
falsified：是否为 OOC 构造样本。
```

如果：

```text
id == image_id 且 falsified=false
```

表示图文来自同一个 VisualNews 记录，通常是 Non-OOC。

如果：

```text
id != image_id 且 falsified=true
```

表示当前 caption 来自一个新闻记录，图片来自另一个新闻记录，是构造出的 OOC 样本。

### 2.2 VisualNews 提供什么

VisualNews 的 `origin/data.json` 提供图片原始上下文。我们本地路径是：

```text
E:\OOC_Datasets\VisualNews\origin\data.json
```

其中一条记录类似：

```json
{
  "id": 39136,
  "caption": "Candace Pickens and her son Zachaeus",
  "topic": "law_crime",
  "source": "washington_post",
  "image_path": "./washington_post/images/0376/501.jpg",
  "article_path": "./washington_post/articles/39136.txt"
}
```

我们用 `id` 或 `image_id` 去回查 VisualNews，得到：

```text
current_caption = VisualNews[id].caption
true_image_context = VisualNews[image_id].caption
```

### 2.3 当前代码怎么实现

入口脚本：

```text
scripts/context/build_cove_lite_context_pairs.py
```

关键逻辑：

```text
1. 扫描 NewsCLIPpings data 目录下所有 JSON。
2. 读取 VisualNews origin/data.json，构建 id -> context 的索引。
3. 若 origin/data.json 可用，优先使用它。
4. articles_metadata/*.p 作为 fallback，因为里面多是 id/article_path，不一定有 caption 文本。
5. 对每条 NewsCLIPpings 样本：
   - sample_id = id
   - image_id = image_id
   - current_caption = meta[id].context
   - true_image_context = meta[image_id].context
   - label = falsified 转成 0/1
6. 输出 JSONL。
```

输出文件：

```text
outputs/cove_lite_context_pairs.jsonl
outputs/cove_lite_context_pairs.jsonl.stats.json
```

输出样本结构：

```json
{
  "sample_id": "213594",
  "image_id": "318944",
  "text_id": "213594",
  "split": "train",
  "domain": "usa_today",
  "label": 1,
  "current_caption": "The opening page of the Community Health Systems web site...",
  "true_image_context": "Louisiana based Lamar Advertising has donated...",
  "source": "visualnews_metadata"
}
```

### 2.4 当前产生了什么效果

你的 smoke test 已经显示：

```text
NewsCLIPpings JSON files = 15
VisualNews origin/data.json records = 1,259,732
COVE-lite 输出 records = 200
missing_text = 0
missing_true_context = 0
```

这说明：

```text
1. 路径正确。
2. NewsCLIPpings 与 VisualNews id 可以对齐。
3. true_image_context 能构造出来。
4. 后续归因不再是空输入。
```

### 2.5 第一轮追问：为什么不用 articles_metadata 里的 pickle？

答：我们检查后发现 `articles_metadata/*.p` 很多记录只包含：

```text
title_len_w, ori_id, image_id, article_path, article_id, sen_num, total_words, id
```

这些字段不一定包含真正 caption/title 文本。直接从这里抽 true context 会导致大量 `missing_true_context`。

所以修正为：

```text
优先读取 VisualNews/origin/data.json；
articles_metadata/*.p 只作为 fallback。
```

这是一次关键修复，否则 COVE-lite 会有严重上下文缺失。

### 2.6 第二轮追问：COVE-lite 和 COVE 有什么区别？

答：COVE 的核心思想是 context-first：先获取或预测图片真实上下文，再判断当前 caption 是否和真实上下文一致。

我们不是完整复现 COVE。我们的版本叫 COVE-lite：

```text
完整 COVE：可能涉及图像上下文检索/生成/验证。
我们的 COVE-lite：利用 NewsCLIPpings 和 VisualNews 的已知数据关系，直接通过 image_id 取 VisualNews 原始 caption 作为 true context。
```

所以答辩口径是：

```text
我们迁移 COVE 的 context-first 思路，但没有声称实现完整 COVE 模型。
```

### 2.7 进一步改进

数据层可以继续增强：

```text
1. true context 不只用 caption，还可以拼接 article title、topic、source、article 首句。
2. 对 article_path 读取文章正文，构造更丰富 evidence。
3. 做 context coverage 统计：不同 source/domain 下 image_id 对齐率。
4. 对 true context 长度、字段覆盖率做质量分层。
5. 增加缓存，避免每次脚本重新扫 356MB data.json 和多个 pickle。
```

---

## 3. VDT 主分类层：为什么保留 VDT，不让解释模块改分类

### 3.1 VDT 在系统里的角色

VDT 是 baseline 主分类器。它回答：

```text
image + current_caption 是否 OOC？
```

我们已经完成两组 strict BLIP-2/GaussianBlur baseline 复现：

| Setting | F1 | Acc | AUC | 状态 |
|---|---:|---:|---:|---|
| bbc,guardian bs128 ep20 | 0.7353 | 0.7383 | 0.7398 | completed |
| usa_today,washington_post bs64 ep20 | 0.8032 | 0.8032 | 0.8028 | completed |
| usa_today,washington_post bs128 ep20 | - | - | - | CUDA OOM |

### 3.2 为什么解释模块不覆盖 VDT label

因为用户明确要求分类准确率不能降低。最稳的设计是：

```text
classification_label = VDT_label
attribution_output = sidecar_output
```

也就是说：

```text
VDT 决定是不是 OOC；
VDT-COVE-Attr 只解释可能错在哪里。
```

如果解释模块出错，它不会影响主分类结果。

### 3.3 当前代码怎么体现

演示 pipeline 中有 baseline-preserving 思路，相关文件包括：

```text
src/e3vdt/inference/cove_attr_pipeline.py
scripts/check_accuracy_preserving.py
```

实验脚本中，v2 attribution 也是单独输出：

```text
outputs/field_nli_attribution_v2.jsonl
```

字段为：

```json
{
  "v2_mismatch_type": "...",
  "v2_conflict_fields": [...],
  "field_nli": {...},
  "evidence_relevance": {...},
  "graph_alignment": {...}
}
```

它不直接写回 VDT checkpoint 或 VDT classifier。

### 3.4 第一轮追问：这样是否只是后处理？

答：是后置 attribution sidecar。我们应该承认它是后处理解释模块，而不是新的端到端主分类器。

但这不是缺点，因为本项目目标不是超过 VDT 分类准确率，而是在保持 VDT 分类能力的同时补充可解释性。

准确表述：

```text
我们提出的是 VDT-based OOC detection 的 evidence-grounded attribution extension。
```

不要说：

```text
我们提出了一个全面超过 VDT 的新 OOC 分类模型。
```

### 3.5 第二轮追问：如果解释模块发现冲突，但 VDT 判断 Non-OOC，怎么办？

答：当前系统有两个输出层：

```text
1. 主分类层：仍然尊重 VDT label。
2. 解释层：可以提示字段冲突或证据不足。
```

如果出现：

```text
VDT_label = Non-OOC
sidecar_conflict_fields = [location]
```

我们不直接改成 OOC，而是把它作为：

```text
attribution warning / evidence warning
```

后续若要融合，需要在验证集上证明不会降低分类 F1/Acc。

### 3.6 进一步改进

后续可做三种融合实验：

```text
1. baseline-preserving：完全不改 VDT label。
2. warning-only：VDT label 不变，但显示 sidecar warning。
3. calibrated fusion：只有当 NLI contradiction 高置信且 evidence sufficient 时，才调整分类分数。
```

必须在验证集上比较：

```text
VDT baseline vs fusion
Acc / F1 / AUC 是否下降
```

如果下降，就不能采用融合。

---

## 4. 事件字段抽取层：如何从文本变成 event tuple

### 4.1 这一步输入输出是什么

输入两段文本：

```text
current_caption
true_image_context
```

输出两个事件结构：

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

这一步的目标是：把自然语言变成可比较字段。

### 4.2 当前实现文件

核心文件：

```text
src/e3vdt/event/enhanced_extractor.py
src/e3vdt/event/extractor.py
src/e3vdt/event/normalize.py
scripts/event/extract_event_tuples_v2.py
```

`extract_event_tuples_v2.py` 对每条 context pair 做：

```python
cur_event = event_from_text(current_caption, extractor="enhanced")
true_event = event_from_text(true_image_context, extractor="enhanced")
```

然后输出：

```text
outputs/event_tuples_v2.jsonl
outputs/event_tuples_v2.jsonl.stats.json
```

### 4.3 当前 enhanced extractor 由哪些部分组成

#### 4.3.1 Rule fallback

`src/e3vdt/event/extractor.py` 是旧规则抽取器。

它主要做：

```text
1. 正则抽时间。
2. 词表/后缀抽地点。
3. 关键词抽 event_type。
4. 关键词抽 relation。
5. 大写短语/中文后缀抽 entity。
```

它的作用是：

```text
无依赖、可复现、作为 baseline。
```

但它不是最终高标准方法。

#### 4.3.2 spaCy NER

`enhanced_extractor.py` 中的 `extract_with_spacy()` 会尝试加载：

```text
en_core_web_sm
```

映射关系：

```text
PERSON / ORG / NORP -> entities
GPE / LOC / FAC -> locations
DATE / TIME -> times
```

如果 spaCy 或模型不可用：

```text
返回空字段，不让 pipeline 崩溃。
```

这让系统具有 fallback 能力。

#### 4.3.3 OpenIE-like triples

`extract_openie_like_triples(text)` 实现轻量关系三元组抽取。

当前支持常见新闻句式：

```text
主体 + 动词 + 宾语
主体 + in/at/near + 地点
```

输出：

```json
{
  "subject": "Police",
  "predicate": "arrested",
  "object": "demonstrators near the court"
}
```

注意：它不是完整 OpenIE/SRL。应该说：

```text
OpenIE/SRL-like lightweight fallback relation extractor。
```

#### 4.3.4 归一化

`normalize.py` 做：

```text
1. lowercase
2. 去标点
3. 去重
4. location alias
5. event type alias
```

例如：

```text
U.S. / US / USA / America -> united states
UK / Britain / England -> united kingdom
war / attack / strike -> war/conflict
rally / demonstration / march -> protest
```

### 4.4 当前产生了什么效果

你的 smoke test 中：

```text
event_tuples_v2 records = 200
current_entities = 200
current_locations = 121
current_times = 61
true_entities = 200
true_locations = 130
true_times = 62
current_event_types = 61
true_event_types = 56
current_relations = 71
true_relations = 69
```

说明：

```text
1. 每条样本基本都抽出了实体。
2. 地点、时间、事件类型、关系都有一定覆盖。
3. relations 字段不是空的，graph alignment 有输入基础。
```

### 4.5 第一轮追问：为什么要抽五类字段？

答：因为 OOC 误用通常发生在这些可解释字段上。

```text
entity：图片里的人或组织被换了。
location：图片地点被换了。
time：旧图新用。
event_type：同一图片被放到不同类型事件中。
relation：同一实体但行为关系不同。
```

这五类字段能覆盖大部分新闻语境错配。

### 4.6 第二轮追问：为什么不直接比较整句相似度？

答：整句相似度无法回答“错在哪里”。

例如：

```text
A protest happened in Paris in 2024.
A protest happened in London in 2024.
```

整句很相似，但错在 location。

再如：

```text
Biden visited Ukraine.
Biden met officials in Washington.
```

都包含 Biden，但关系和地点不同。

所以必须拆成字段，否则无法输出可解释错配类型。

### 4.7 第三轮追问：当前抽取器有什么不足？

不足包括：

```text
1. spaCy 小模型对复杂新闻实体不一定准。
2. 中文/多语言支持有限。
3. OpenIE-like triples 是正则模板，不是真正 SRL。
4. 时间没有完整标准化。
5. 地名 alias 词典很小。
6. event_type taxonomy 比较粗。
```

所以当前应叫：

```text
enhanced extractor baseline
```

不能说是完整事件抽取模型。

### 4.8 进一步改进

可以分三步：

```text
短期：扩展 location/event aliases，增加 dateparser 时间标准化。
中期：接入 Stanza / OpenIE / SRL，把 relations_structured 换成真实谓词-论元结构。
长期：使用 AMR parser 或 LLM JSON extractor，输出 trigger/argument/role schema。
```

需要补的实验：

```text
人工标注 100-300 条 event tuple，评估：
Entity F1
Location F1
Time F1
Event Type F1
Relation F1
Exact Tuple Match
```

---

## 5. Evidence relevance 层：为什么证据不足时不能强行解释

### 5.1 这一步解决什么问题

有些 true_image_context 很短或信息不足。例如：

```text
A file photo.
People at an event.
```

这种上下文不足以判断：

```text
地点是否错？
时间是否错？
主体是否错？
```

如果系统仍然强行输出 `location mismatch`，就是不可靠解释。

### 5.2 当前实现文件

```text
src/e3vdt/attribution/evidence_relevance_v2.py
```

调用位置：

```text
scripts/attribution/run_field_nli_attribution_v2.py
```

每条样本会调用：

```python
evidence = score_evidence_relevance(
    current_caption,
    true_context,
    cur_event,
    true_event
)
```

### 5.3 当前怎么算 relevance

代码计算四部分：

```text
1. text_similarity(current_caption, true_context)
2. field_overlap_mean(entity/location/time/event_type/relation)
3. context_len_score
4. information_score
```

公式：

```text
relevance =
  0.35 * text_similarity
+ 0.35 * field_overlap_mean
+ 0.15 * context_len_score
+ 0.15 * information_score
```

其中：

```text
text_similarity：SequenceMatcher 字符串相似度。
field_overlap：五类事件字段集合 overlap。
context_len_score：true_context 长度是否足够。
information_score：true_event 中非空字段数。
```

默认判定条件：

```text
true_context 长度 >= 20
true_event 至少有 1 个非空字段
relevance >= 0.25
```

满足则：

```json
{
  "evidence_sufficiency": "sufficient"
}
```

否则：

```json
{
  "evidence_sufficiency": "insufficient",
  "evidence_reason": "true_context_too_short / true_context_has_no_extracted_fields / low_relevance"
}
```

### 5.4 当前产生什么效果

它让系统具备“拒绝解释”的能力：

```text
如果证据不足，输出 uncertain / evidence insufficient。
```

最终决策中优先判断：

```python
if evidence_sufficiency != "sufficient":
    mismatch_type = "uncertain / evidence insufficient"
    conflict_fields = ["evidence_insufficient"]
```

### 5.5 第一轮追问：为什么需要 evidence relevance？

答：因为事实核验系统不能默认所有外部上下文都是好证据。

在 OOC 检测中，如果 true context 本身信息不足或抽取失败，强行解释会产生假归因。

所以 evidence relevance 的必要性是：

```text
1. 防止噪声证据误导解释。
2. 防止上下文字段缺失导致错判。
3. 让系统知道什么时候应该说“不能可靠判断”。
```

### 5.6 第二轮追问：当前 relevance 是不是还是规则？

答：是。当前实现是 RED-DOT/CMIE 思想的轻量启发式版本，不是完整训练模型。

应该说：

```text
我们实现了 evidence relevance / sufficiency gate 的可运行版本，当前采用文本相似度、字段 overlap、上下文长度和信息量组合打分。
```

不能说：

```text
我们复现了 RED-DOT 或 CMIE 模型。
```

### 5.7 进一步改进

可以逐步升级：

```text
1. 使用 NLI contradiction/entailment 分数参与 relevance。
2. 训练二分类 evidence relevance classifier。
3. 引入检索证据，多条 evidence 做 rerank。
4. 做 ablation：无 relevance vs length-only vs overlap vs NLI+overlap。
5. 在人工 gold set 上看 gate 是否减少错误解释。
```

---

## 6. Field-wise NLI 层：你现在正式跑 NLI 到底在做什么

### 6.1 NLI 的输入输出是什么

NLI 是 Natural Language Inference，自然语言推理。

它判断：

```text
premise 是否支持 hypothesis？
```

输出三类：

```text
entailment：支持
neutral：无法判断
contradiction：矛盾
```

在我们系统里：

```text
premise = true_image_context
hypothesis = current_caption 中抽出的字段声明
```

### 6.2 当前实现文件

```text
src/e3vdt/attribution/field_nli_v2.py
scripts/attribution/run_field_nli_attribution_v2.py
```

使用的模型：

```text
facebook/bart-large-mnli
```

运行命令里：

```powershell
-NliModel facebook/bart-large-mnli
-NliDevice 0
```

含义是：

```text
用 GPU 0 跑 BART-MNLI 推理。
```

这不是训练，只是推理。

### 6.3 字段如何变成 hypothesis

代码里 `build_field_hypothesis()` 做：

```text
entity -> The image event involves {values}.
location -> The image event took place in {values}.
time -> The image event happened at {values}.
event_type -> The image event is about {values}.
relation -> The image event includes the relation or action: {values}.
```

例子：

```text
current_caption: Protesters gathered in Paris on Monday.
true_context: People gathered in London during a climate demonstration in 2020.
```

抽取字段：

```text
location = Paris
time = Monday
event_type = protest
```

构造 hypothesis：

```text
The image event took place in Paris.
The image event happened at Monday.
The image event is about protest.
```

NLI 判断 true_context 是否支持这些句子。

### 6.4 决策阈值是什么

默认：

```text
contradiction_threshold = 0.60
entailment_threshold = 0.60
```

如果：

```text
contradiction >= 0.60
```

则字段冲突：

```text
field label = contradiction
field 加入 conflict_fields
```

如果：

```text
entailment >= 0.60
```

则字段一致：

```text
field label = entailment
```

否则：

```text
field label = neutral
```

如果多个字段 contradiction，则选择 contradiction 分数最高的字段映射成主 mismatch_type。

### 6.5 当前产生什么输出

输出在：

```text
outputs/field_nli_attribution_v2.jsonl
outputs/field_nli_attribution_v2.jsonl.stats.json
```

单条结构包括：

```json
{
  "v2_mismatch_type": "location mismatch",
  "v2_conflict_fields": ["location"],
  "field_nli": {
    "location": {
      "label": "contradiction",
      "scores": {
        "entailment": 0.02,
        "neutral": 0.10,
        "contradiction": 0.88,
        "_backend": "facebook/bart-large-mnli"
      },
      "hypothesis": "The image event took place in Paris."
    }
  }
}
```

正式 NLI 跑完后要确认：

```text
used_transformers = true
nli_backend_counts 中是 facebook/bart-large-mnli
```

如果是：

```text
lexical_fallback
```

说明没有真正用 transformer NLI，只是 fallback，不能当正式 NLI 实验。

### 6.6 第一轮追问：NLI 为什么比字符串相似度更合理？

答：字符串相似度只看表面字符，无法判断语义支持或矛盾。

例如：

```text
US vs United States
```

字符串不完全一样，但语义一致。

又如：

```text
Paris vs London
```

字符串不同，语义也冲突。

NLI 的优势是：

```text
它用预训练语义模型判断 true context 是否支持字段声明。
```

所以它更适合判断：

```text
当前 caption 的某个字段是否被图片真实上下文反驳。
```

### 6.7 第二轮追问：NLI 是否一定更好？

答：不一定。必须实验验证。

NLI 可能有问题：

```text
1. 新闻 caption 太短，premise 信息不足。
2. hypothesis 模板可能不自然。
3. BART-MNLI 是通用 NLI，不是新闻 OOC 专用。
4. 多实体字段拼在一句 hypothesis 里可能让模型混淆。
5. 时间和地点的细粒度冲突不一定总被 NLI 正确识别。
```

所以我们不能直接说 NLI 一定优于规则。

正确说法是：

```text
我们把 field-wise NLI 作为主方法候选，并用人工 attribution gold set 与 rule/similarity baseline 比较。
```

### 6.8 第三轮追问：为什么 smoke test 全是 benign？

答：因为 smoke test 用了 `-NoTransformers`。

这时：

```text
used_transformers = false
backend = lexical_fallback
```

fallback 只根据词面 overlap 给很保守的分数，基本不会产生高 contradiction，所以容易全判 benign。

它只用于验证链路，不用于正式结果。

正式结果必须跑：

```powershell
-NliModel facebook/bart-large-mnli -NliDevice 0
```

### 6.9 进一步改进

可以改进：

```text
1. 每个字段每个值单独构造 hypothesis，而不是多个值拼一起。
2. 使用更适合事实核验的 NLI 模型。
3. 在人工 dev set 上调 contradiction_threshold。
4. 对 entity/location/time 采用专门的 normalization + NLI 双重判断。
5. 训练轻量 attribution classifier，输入 NLI 分数和 graph/evidence 特征。
```

---

## 7. Graph alignment-lite 层：为什么需要关系级比较

### 7.1 为什么字段级还不够

有些错配不是单个字段能看出来，而是关系组合错了。

例如：

```text
current_caption: Biden visited Ukraine.
true_context: Biden met officials in Washington.
```

单看 entity：

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

### 7.2 当前实现文件

```text
src/e3vdt/attribution/graph_alignment_v2.py
```

调用位置：

```text
scripts/attribution/run_field_nli_attribution_v2.py
```

### 7.3 当前图怎么构造

图不是复杂图数据库，而是由 `relations_structured` 三元组构成。

三元组来自：

```text
extract_openie_like_triples(text)
```

结构：

```json
{
  "subject": "Biden",
  "predicate": "visited",
  "object": "Ukraine"
}
```

可以看成：

```text
node: Biden, Ukraine
edge: visited
```

### 7.4 当前怎么对齐

对每条 current triple，遍历所有 true triple，计算：

```text
subject similarity
predicate similarity
object similarity
```

总分：

```text
score = (subj_sim + pred_sim + obj_sim) / 3
```

找到最高分 true triple，写入：

```json
{
  "current": {...},
  "true": {...},
  "score": 0.42
}
```

如果满足：

```text
subject similarity >= 0.65
predicate similarity >= 0.45
object similarity < 0.35
```

则认为有关系/对象 grounding conflict：

```text
graph_conflicts = ["relation"]
```

### 7.5 当前产生什么效果

每条 v2 输出里有：

```json
{
  "graph_alignment": {
    "graph_alignment_score": 0.42,
    "graph_conflicts": ["relation"],
    "aligned_edges": [...],
    "num_current_edges": 1,
    "num_true_edges": 1
  }
}
```

如果 field-wise NLI 没有发现 contradiction，但 graph alignment 发现 relation conflict，最终决策会把 relation 加入 conflict_fields。

### 7.6 第一轮追问：这是 Evidence-GNN 吗？

答：不是。

当前是：

```text
Evidence-GNN 思路的 lightweight graph matching baseline。
```

它没有训练 GNN，没有 attention-based graph encoder。

它只做：

```text
三元组抽取 -> 字符串相似度对齐 -> 关系冲突启发式判断
```

### 7.7 第二轮追问：那它有什么必要？

答：它建立了 relation-level attribution 的接口。

即使当前是轻量版，它也让系统具备：

```text
1. 保存 relations_structured。
2. 输出 aligned_edges。
3. 输出 graph_conflicts。
4. 支持后续把 graph matching 替换成真正 Evidence-GNN。
```

这比完全没有关系结构要好。

### 7.8 进一步改进

后续升级路线：

```text
1. 用 OpenIE/SRL/AMR 替换正则三元组。
2. 用 sentence embedding 比较 subject/predicate/object。
3. 构造 claim graph 和 evidence graph。
4. 训练 GNN 或 graph attention module。
5. 把 graph score 与 NLI/evidence features 融合。
```

需要补实验：

```text
无 graph alignment vs 有 graph alignment
relation mismatch 子集上的 Field-F1
hard negative 中 same-entity different-relation 的识别率
```

---

## 8. 最终归因决策层：如何从多个模块合成 mismatch_type

### 8.1 输入是什么

最终决策使用：

```text
evidence_relevance
field_nli
graph_alignment
```

当前函数：

```text
scripts/attribution/run_field_nli_attribution_v2.py::final_decision
```

### 8.2 决策顺序

代码逻辑：

```text
第一优先级：证据是否 sufficient。
如果 insufficient，直接输出 uncertain / evidence insufficient。

第二优先级：field-wise NLI contradiction。
如果有 contradiction，把 contradiction 字段作为 conflict_fields。

第三优先级：graph_conflicts。
如果 graph alignment 发现 relation conflict，加入 conflict_fields。

第四优先级：如果没有冲突，输出 benign illustrative image。
```

伪代码：

```python
if evidence_sufficiency != "sufficient":
    mismatch_type = "uncertain / evidence insufficient"
    conflict_fields = ["evidence_insufficient"]

else:
    conflicts = field_conflicts + graph_conflicts
    if conflicts:
        mismatch_type = map_primary_field_to_type(conflicts[0])
    else:
        mismatch_type = "benign illustrative image"
```

### 8.3 字段到类型如何映射

```text
entity -> entity mismatch
location -> location mismatch
time -> temporal mismatch
event_type -> event-type mismatch
relation -> relation mismatch
```

### 8.4 第一轮追问：为什么 evidence insufficiency 优先级最高？

答：因为没有足够证据时，后面的 NLI 和 graph 判断都不可信。

如果 true context 只有：

```text
A file photo.
```

即使 current caption 有 Paris，系统也不能说 location mismatch。它应该说：

```text
证据不足，无法可靠判断。
```

### 8.5 第二轮追问：为什么没有冲突时输出 benign？

答：当前逻辑里，如果 evidence sufficient 且没有字段 contradiction / graph conflict，就认为没有发现语境错配证据，输出：

```text
benign illustrative image
```

但这个输出不代表绝对真实，只代表当前归因模块没有找到冲突。

更严谨的答辩说法：

```text
在当前 evidence 和 extractor/NLI 能力下，没有检测到字段级矛盾。
```

### 8.6 进一步改进

决策层可以升级：

```text
1. 不只用硬阈值，改成学习型 attribution classifier。
2. 引入 confidence calibration。
3. 对 OOC/Non-OOC 分开设置阈值。
4. 支持 multi-label mismatch_type。
5. 对 evidence_insufficient、context_omission、benign illustrative image 做更细分类。
```

---

## 9. 实验链路：一键脚本到底做了什么

### 9.1 入口脚本

```text
scripts/run_vdt_cove_attr_v2_experiments.ps1
```

正式命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vdt_cove_attr_v2_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -Python python `
  -MaxRecords 500 `
  -EvalSampleN 80 `
  -NliModel facebook/bart-large-mnli `
  -NliDevice 0
```

### 9.2 0/7 input check

运行：

```text
scripts/diagnose_cove_lite_inputs.py
```

检查：

```text
NewsCLIPpings JSON 数量
VisualNews pickle 数量
VisualNews origin/data.json 是否存在
样例 id/image_id/falsified
```

输出：

```text
outputs/input_check.json
```

### 9.3 1/7 COVE-lite true context pairs

运行：

```text
scripts/context/build_cove_lite_context_pairs.py
```

输出：

```text
outputs/cove_lite_context_pairs.jsonl
outputs/cove_lite_context_pairs.jsonl.stats.json
```

关键字段：

```text
total
available_before_cap
kept
missing_text
missing_true_context
```

### 9.4 2/7 v1 weak rule labels

运行：

```text
scripts/labels/build_weak_attribution_from_context.py
```

它使用旧规则 sidecar 生成 baseline 归因：

```text
weak_mismatch_type
weak_conflict_fields
event_scores
```

输出：

```text
outputs/weak_attribution_labels.jsonl
```

它是 baseline，不是最终主方法。

### 9.5 3/7 v2 event tuple extraction

运行：

```text
scripts/event/extract_event_tuples_v2.py
```

输出：

```text
outputs/event_tuples_v2.jsonl
outputs/event_tuples_v2.jsonl.stats.json
```

新增字段：

```text
current_event
true_event
event_extractor
```

### 9.6 4/7 v2 NLI + evidence + graph

运行：

```text
scripts/attribution/run_field_nli_attribution_v2.py
```

输出：

```text
outputs/field_nli_attribution_v2.jsonl
outputs/field_nli_attribution_v2.jsonl.stats.json
```

关键检查：

```text
records = 500
used_transformers = true
nli_backend_counts 包含 facebook/bart-large-mnli
```

### 9.7 5/7 build manual annotation candidates

运行：

```text
scripts/eval/build_attribution_eval_sample.py
```

输出：

```text
examples/attribution_eval_candidates.jsonl
```

这是给队友人工标注的候选集。

每条需要填：

```text
gold_mismatch_type
gold_conflict_fields
annotator
rationale
annotation_status = done
```

### 9.8 6/7 evaluate if gold set exists

如果存在：

```text
examples/attribution_eval_set.jsonl
```

且 sample_id 与当前预测有交集，则运行：

```text
scripts/eval/evaluate_attribution_v2.py
```

输出：

```text
outputs/attribution_eval_v2_metrics.json
outputs/attribution_eval_v2_metrics.csv
```

如果没有交集，脚本会跳过，避免生成 `matched=0` 的假指标。

### 9.9 7/7 collect tables

运行：

```text
scripts/collect_v2_report_tables.py
```

输出：

```text
outputs/report_tables_v2.md
```

里面汇总：

```text
VDT 复现表
COVE-lite coverage
event extraction coverage
v2 attribution distribution
manual gold evaluation
```

### 9.10 第一轮追问：为什么要先跑 NoTransformers smoke test？

答：因为正式 NLI 会下载/加载大模型，耗时更长。先用：

```powershell
-NoTransformers
```

能快速验证：

```text
路径是否对
数据是否非空
COVE-lite 是否能对齐
事件抽取是否正常
候选集是否能生成
```

但 NoTransformers 结果不能作为正式 NLI 结果。

### 9.11 第二轮追问：正式 NLI 跑完看什么？

看：

```powershell
Get-Content outputs\field_nli_attribution_v2.jsonl.stats.json
```

必须确认：

```text
used_transformers = true
nli_backend_counts 不是 lexical_fallback
records = MaxRecords
```

再看：

```powershell
Get-Content examples\attribution_eval_candidates.jsonl -TotalCount 5
Get-Content outputs\report_tables_v2.md
```

确认 candidates 和 report table 正常。

---

## 10. 人工归因评估：为什么最终结论必须靠 gold set

### 10.1 为什么需要人工标注

NewsCLIPpings 只有二分类标签：

```text
falsified / not falsified
```

没有：

```text
mismatch_type
conflict_fields
```

所以系统输出的错配类型必须通过人工 gold set 验证。

### 10.2 标注对象是什么

候选文件：

```text
examples/attribution_eval_candidates.jsonl
```

每条包括：

```text
current_caption
true_image_context
label
weak_mismatch_type
weak_conflict_fields
v2_mismatch_type
v2_conflict_fields
```

人工要填：

```json
{
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location", "time"],
  "annotator": "A",
  "rationale": "current caption says Paris/Monday, true context says London/2020",
  "annotation_status": "done"
}
```

### 10.3 评估指标是什么

`evaluate_attribution_v2.py` 计算：

```text
mismatch_type_accuracy
conflict_field_micro_precision
conflict_field_micro_recall
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

对比方法包括：

```text
v2_field_nli_evidence_graph
weak_rule_sidecar
```

后续还可以加：

```text
majority baseline
random baseline
similarity-only baseline
MUSE-lite baseline
```

### 10.4 第一轮追问：为什么不能用 weak labels 当 gold？

答：weak labels 是系统自己用规则生成的，不是真实答案。

如果用 weak labels 当 gold，就等于：

```text
用规则评价规则，结果没有说服力。
```

所以 weak labels 只能用于：

```text
1. 候选样本预分层。
2. baseline 对比。
3. 辅助人工标注。
```

不能当最终 gold。

### 10.5 第二轮追问：人工标多少条才够？

课程项目最低可先做：

```text
50-100 条
```

更稳妥：

```text
100-300 条
```

要尽量覆盖：

```text
Non-OOC
entity mismatch
location mismatch
temporal mismatch
event-type mismatch
relation mismatch
evidence insufficient
multi-field conflict
```

### 10.6 进一步改进

人工评估可以升级：

```text
1. 双人标注，计算一致性。
2. 冲突样本仲裁。
3. 分 source/domain 统计。
4. 分 hard negative 类型统计。
5. 使用标注集调 NLI 阈值。
```

---

## 11. Gradio 展示系统：答辩时如何演示

### 11.1 入口

```text
demo/app.py
```

启动：

```powershell
python demo/app.py
```

或：

```powershell
.\scripts\start_demo.ps1
```

### 11.2 展示哪些 tab

主要展示：

```text
VDT-COVE-Attr 主系统
分类不降验证
复现实验指标
实验看板
```

### 11.3 推荐演示路线

```text
1. 打开首页，说明系统模块。
2. 进入 VDT-COVE-Attr 主系统。
3. 展示 route01_location_time：地点/时间错配。
4. 展示 route03_entity：主体错配。
5. 展示 route06_evidence_insufficient：证据不足不强行解释。
6. 展示 route08_multi_field：多字段冲突。
7. 切到分类不降验证，说明 sidecar 不覆盖 VDT label。
8. 切到复现实验指标，展示 VDT strict baseline。
9. 切到实验看板，说明人工 gold set 和 ablation 正在补。
```

### 11.4 第一轮追问：网页 demo 和正式实验有什么区别？

答：网页 demo 是系统验收和流程展示，使用 curated cases 验证输入输出和解释结构。

正式实验是：

```text
NewsCLIPpings / VisualNews 样本
+ BART-MNLI 推理
+ 人工 gold set
+ 指标评估
```

不能把 demo case 的 1.0 smoke 指标当最终实验结果。

### 11.5 进一步改进

展示系统可以继续增强：

```text
1. 支持上传真实图片并调用 VDT checkpoint。
2. 显示 field-wise NLI score bar。
3. 显示 graph aligned edges 可视化。
4. 显示 evidence sufficiency reason。
5. 支持导出单例报告。
```

---

## 12. 当前系统完整流程复述版

答辩时可以这样完整讲：

> 我们的项目基于 NewsCLIPpings / VisualNews 做 out-of-context image misinformation detection。VDT baseline 已经负责判断 image-caption 是否 OOC，但它不能解释具体错在哪里。因此我们设计了 VDT-COVE-Attr attribution sidecar。  
>   
> 首先，系统读取 NewsCLIPpings 样本，其中 `id` 表示当前 caption 来源，`image_id` 表示当前图片来源。我们用这两个 id 回查 VisualNews 的 `origin/data.json`，得到 current caption 和图片原始 true image context。这个步骤叫 COVE-lite，因为它迁移了 COVE 的 context-first 思路，但不是完整 COVE 模型。  
>   
> 第二步，我们分别对 current caption 和 true image context 做 enhanced event extraction，抽取 entity、location、time、event_type、relation 五类字段，并用轻量 OpenIE-like fallback 抽取 subject-predicate-object 三元组。当前实现包含规则 fallback、spaCy NER、字段归一化和关系三元组接口。  
>   
> 第三步，我们做 evidence relevance / sufficiency 判断。如果 true context 太短、抽不出有效字段，或者 relevance 太低，系统输出 evidence insufficient，不强行解释。  
>   
> 第四步，我们做 field-wise NLI。对 current caption 中抽出的每个字段构造 hypothesis，例如 location 字段构造成 “The image event took place in Paris.”，然后把 true image context 作为 premise，用 `facebook/bart-large-mnli` 判断 entailment、neutral 或 contradiction。如果某个字段 contradiction 分数超过阈值，就把它作为 conflict field。  
>   
> 第五步，我们做 graph alignment-lite。它读取 OpenIE-like 三元组，对 current caption 和 true context 的 subject、predicate、object 分别计算相似度。如果主体和行为相似但对象明显不同，就标记 relation conflict。当前这不是 Evidence-GNN，而是 Evidence-GNN 思路的轻量 graph matching baseline。  
>   
> 最后，系统综合 evidence sufficiency、field-wise NLI conflicts 和 graph conflicts 输出 mismatch_type、conflict_fields 和 explanation。整个 attribution sidecar 不覆盖 VDT 主分类 label，因此不会降低 VDT baseline 的分类准确率。归因模块是否真正可靠，我们通过人工 attribution gold set 计算 Type Accuracy、Field Micro-F1、Macro-F1 和 Exact Match Rate 来验证。

---

## 13. 老师可能追问与回答

### Q1：你们的创新点到底是什么？

答：不是重新提出一个比 VDT 更强的主分类器，而是在 VDT 二分类基础上提出 evidence-grounded attribution extension。我们把 OOC 解释拆成 true context 构造、事件字段抽取、证据充分性判断、字段级 NLI 矛盾检测和关系图对齐，并用人工 gold set 评估解释可靠性。

### Q2：你们是不是只是加规则？

答：规则模块只作为 baseline 和 fallback。主方法候选是 COVE-lite true context + field-wise NLI + evidence relevance + graph alignment-lite。当前 OpenIE 和 graph 部分确实是 lightweight 实现，我们不会说成完整 OpenIE/SRL 或 Evidence-GNN。后续会用人工 gold set 验证 NLI 版是否优于规则版。

### Q3：为什么用 VisualNews caption 当 true context？

答：NewsCLIPpings 来自 VisualNews，OOC 样本通过重新配对 caption 和 image 构造。`image_id` 对应图片在 VisualNews 中的原始记录，因此其原始 caption 是图片真实语境的直接 evidence。这是 COVE-lite 的数据基础。

### Q4：NLI 在这里做什么？

答：NLI 判断 true image context 是否支持或反驳 current caption 中的字段声明。例如 current caption 声称地点是 Paris，我们构造 hypothesis “The image event took place in Paris.”，然后用 true context 判断 entailment/neutral/contradiction。如果 contradiction 高，就输出 location mismatch。

### Q5：为什么要 evidence relevance？

答：因为不是所有 true context 都足以解释。如果上下文太短或抽不到字段，系统不能强行说某个字段冲突，而应该输出 evidence insufficient。这能减少错误解释。

### Q6：Graph alignment 是不是 GNN？

答：不是。当前是 lightweight graph matching，用 OpenIE-like triples 做 subject-predicate-object 对齐。它是 Evidence-GNN 思路的可运行 baseline，不是训练好的 GNN。

### Q7：你们怎么证明解释是对的？

答：必须通过人工 attribution gold set。我们生成候选样本，让人工标 `gold_mismatch_type` 和 `gold_conflict_fields`，再计算 Type Acc、Field Micro-F1、Macro-F1 和 Exact Match。没有人工 gold set 前，不能声称解释模块已经严格有效。

### Q8：会不会降低 VDT 分类准确率？

答：默认不会。因为 attribution sidecar 不覆盖 VDT label。主分类结果仍由 VDT 给出，解释模块只输出错配类型和冲突字段。只有未来在验证集证明融合不降指标时，才考虑把归因信号反向融合进分类。

---

## 14. 当前还缺什么，下一步怎么补

### 14.1 已完成

```text
1. VDT strict baseline 两组复现。
2. Gradio 系统演示闭环。
3. COVE-lite true context 构造。
4. Enhanced event extraction。
5. Field-wise NLI 推理接口。
6. Evidence relevance gate。
7. Graph alignment-lite。
8. v2 一键实验脚本。
9. 人工 annotation candidates 生成脚本。
10. attribution evaluation 脚本。
```

### 14.2 正在进行

```text
正式 NLI 500 条推理。
```

### 14.3 必须补的实验

```text
1. 人工标注 80-100 条 attribution gold set。
2. 评估 weak rule vs v2 NLI。
3. 做 evidence relevance ablation。
4. 做 hard negative 子集分析。
5. 做 event extraction 小规模字段 F1。
```

### 14.4 高质量改进路线

```text
第一阶段：把当前 500 条 NLI 跑完，生成 candidates。
第二阶段：队友分工标注 candidates。
第三阶段：跑 evaluate_attribution_v2.py，得到解释指标。
第四阶段：如果 NLI 不如规则，分析失败原因，调 hypothesis 和阈值。
第五阶段：替换 OpenIE-like fallback 为 SRL/OpenIE/AMR。
第六阶段：做 Evidence-GNN-lite 或 trainable attribution head。
```

---

## 15. 最终口径总结

最稳的总结是：

```text
我们完成了一个 VDT-based OOC detection 的可解释扩展系统。VDT 保持主分类，COVE-lite 根据 VisualNews 构造图片真实语境，enhanced extractor 抽取事件字段，field-wise NLI 判断字段是否被真实语境反驳，evidence relevance 控制证据不足，graph alignment-lite 补充关系冲突。当前系统闭环和演示已经完成，OpenIE/SRL 和 Evidence-GNN 仍是轻量替代版本，归因有效性需要通过人工 gold set 和 ablation 实验验证。
```

这句话既说明做了什么，也不夸大。
