# VDT-CF-Attr 全流程深度讲稿：基于可控反事实训练的图文错配原因解释

> 版本定位：这是对旧版 `VDT-COVE-Attr-模块方法深挖版.md` 的彻底修正。  
> 旧版主线是“用 true image context 与 current caption 比较，然后直接输出错配原因”。  
> 新版主线是：**true context 只用于训练数据构造和评估参考，推理阶段不依赖 true context；模型只输入 image + current caption，在 VDT 判断 OOC 后输出错配原因。**
>
> 推荐新名称：**VDT-CF-Attr**  
> 中文：**基于可控反事实训练的 VDT 图文错配归因方法**  
> 其中 CF = Controlled Counterfactual。

---

## 0. 一句话讲清楚我们现在到底做什么

我们的系统分两层：

```text
第一层：VDT
输入 image + current_caption
输出 OOC / Non-OOC
回答：图文是否错配？

第二层：VDT-CF-Attr
输入 image + current_caption + VDT score
输出 mismatch_type + conflict_fields
回答：如果错配，主要错在哪里？
```

最重要的修正是：

```text
推理阶段不输入 true_image_context。
```

如果推理阶段还要输入 true context，那么系统本质上就是：

```text
current_caption vs true_context 比较
```

这当然可以做解释，但实际应用价值受限，因为真实场景中通常没有图片原始上下文。我们提出可控反事实训练，就是为了让模型在训练阶段利用原始匹配样本构造监督信号，最终在推理阶段只凭 **图片与当前文本的多模态关系** 判断错配原因。

---

## 1. 研究背景：为什么 VDT 还不够

### 1.1 OOC 任务是什么

Out-of-Context misinformation 指的是：图片本身可能是真的，文本也可能是真的，但二者被错误配对，形成误导。例如：

```text
图片：2019 年伦敦抗议现场
文本：2024 年巴黎抗议现场
```

图片和文本都像新闻，但不是同一事件。

### 1.2 VDT 做了什么

VDT 是我们保留的主分类 baseline。它解决的是：

```text
image + caption -> OOC / Non-OOC
```

它关注跨域 OOC 检测，例如不同新闻机构、不同新闻领域下的泛化。VDT 的价值在于主分类性能，而不是细粒度解释。

### 1.3 VDT 没有解决什么

VDT 不能直接告诉我们：

```text
错的是人物？
错的是地点？
错的是时间？
错的是事件类型？
还是整件事都错？
```

所以我们在 VDT 后面做 attribution head。

---

## 2. 旧版路线为什么要修正

旧版路线是：

```text
image + current_caption
  ↓
用 image_id 找 VisualNews true_image_context
  ↓
current_caption vs true_image_context
  ↓
NLI / graph alignment
  ↓
输出 mismatch_type
```

这个路线可以作为 **oracle / upper-bound / baseline**，但不能作为最终系统，原因是：

### 2.1 推理阶段依赖 true context

如果新图片来自数据集外，我们没有 VisualNews `image_id -> original caption` 映射，就无法得到 true context。此时系统不能解释。

### 2.2 解释能力来自外部原始上下文，不是模型自己学到的图文归因

旧版方法更像：

```text
拿当前文本和原始文本比差异
```

而不是：

```text
模型根据图片内容和当前文本判断哪里不一致
```

### 2.3 true context 可以用于训练和评估，但不能作为最终推理输入

新版明确规定：

| 用途 | 是否允许使用 true context |
|---|---:|
| 构造反事实训练样本 | 允许 |
| 人工标注真实 OOC 评测集 | 允许作为参考 |
| 训练模型输入特征 | 不允许作为最终主方法 |
| 推理阶段输入 | 不允许 |
| oracle baseline / upper bound | 允许，但必须单独标注 |

---

## 3. 新版总体路线

新版完整流程分为两个阶段：训练阶段和推理阶段。

### 3.1 训练阶段

```text
原始 non-OOC 匹配样本
(image I, caption T, label=0)
        ↓
抽取 caption 中可编辑字段和 span
entity / location / time
        ↓
只替换一个字段，生成 edited_caption T'
        ↓
得到可控 OOC 负样本
(image I, edited_caption T', mismatch_type = 被替换字段)
        ↓
用 image + edited_caption 提取多模态特征
        ↓
训练 Attribution Head
```

### 3.2 推理阶段

```text
输入 image I + current_caption T
        ↓
VDT 判断 OOC / Non-OOC
        ↓
若 Non-OOC：输出 none / benign
若 OOC：进入 Attribution Head
        ↓
Attribution Head 只基于 image + current_caption 特征
        ↓
输出 mismatch_type + conflict_fields + confidence
```

### 3.3 新版核心主张

```text
反事实样本用于提供细粒度监督标签；
Attribution Head 学习 image-caption 字段冲突模式；
推理时不依赖 true context。
```

---

## 4. 模块 A：可控反事实样本构造

### 4.1 这个模块解决什么问题

原始 NewsCLIPpings 只有二分类标签：

```text
OOC / Non-OOC
```

没有：

```text
location mismatch
temporal mismatch
entity mismatch
```

所以不能直接监督训练解释模块。我们需要自己构造带确定错配原因的样本。

### 4.2 为什么从 non-OOC 样本出发

只选择原本匹配的样本：

```json
{
  "image_id": "...",
  "caption": "Biden spoke in Washington in 2024.",
  "label": 0
}
```

因为只有原本匹配时，我们才能保证：

```text
图片 I 与原始 caption T 是一致的。
```

如果我们把 T 中的一个字段改掉，例如地点：

```text
Biden spoke in Paris in 2024.
```

那么新样本的错配原因就是可控的：

```text
location mismatch
```

原始 OOC 样本不能直接用于细粒度监督，因为它可能同时有人物、地点、时间、事件类型多种冲突。

### 4.3 字段抽取：从 caption 找到可编辑对象

我们先从 caption 中抽取三类字段：

```text
entity
location
time
```

第一版不做 event_type / relation，因为它们更难稳定编辑。

#### 输入

```text
Biden spoke in Washington in 2024.
```

#### 抽取结果

```json
{
  "entities": [
    {"text": "Biden", "type": "PERSON", "start": 0, "end": 5}
  ],
  "locations": [
    {"text": "Washington", "type": "GPE", "start": 15, "end": 25}
  ],
  "times": [
    {"text": "2024", "type": "YEAR", "start": 29, "end": 33}
  ]
}
```

这里必须有 span：

```text
start / end
```

因为我们不是重新生成整句话，而是在原句中做最小替换。

### 4.4 span 为什么重要

如果只知道：

```text
locations = ["Washington"]
```

但不知道它在句子哪里，就很难保证只改一个字段。span 能保证：

```text
原句前半部分 + replacement + 原句后半部分
```

例如：

```text
caption[:15] + "Paris" + caption[25:]
```

得到：

```text
Biden spoke in Paris in 2024.
```

### 4.5 replacement pool 如何建立

replacement pool 不是手写几个词，而是从训练集 non-OOC 样本中自动抽取。

#### entity pool

按类型分组：

```text
PERSON: Biden, Trump, Macron, Sunak
ORG: UN, NATO, WHO
GPE/NORP: United States, France, China
```

替换规则：

```text
PERSON -> PERSON
ORG -> ORG
GPE -> GPE
```

禁止：

```text
Biden -> Paris
UN -> 2024
```

#### location pool

收集地点实体：

```text
Paris, London, Washington, Tokyo, Beijing, Gaza, New York
```

替换规则：

```text
location -> different location
```

如果能区分城市/国家更好：

```text
city -> city
country -> country
```

第一版至少保证 GPE/LOC/FAC 内替换。

#### time pool

第一版只做年份：

```text
2018, 2019, 2020, 2021, 2022, 2023, 2024
```

原因：年份替换稳定、span 清楚、不会破坏语法。

暂时不建议做复杂日期：

```text
Monday -> Friday
January -> March
June 5 -> July 9
```

因为可能引入语义和语法问题。

### 4.6 单字段反事实编辑

每个负样本只改一个字段。

#### location_swap

原句：

```text
Biden spoke in Washington in 2024.
```

替换：

```text
Washington -> Paris
```

结果：

```text
Biden spoke in Paris in 2024.
```

标签：

```json
{
  "label": 1,
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "edit_type": "location_swap"
}
```

#### time_swap

```text
2024 -> 2020
```

结果：

```text
Biden spoke in Washington in 2020.
```

标签：

```json
{
  "label": 1,
  "gold_mismatch_type": "temporal mismatch",
  "gold_conflict_fields": ["time"],
  "edit_type": "time_swap"
}
```

#### entity_swap

```text
Biden -> Trump
```

结果：

```text
Trump spoke in Washington in 2024.
```

标签：

```json
{
  "label": 1,
  "gold_mismatch_type": "entity mismatch",
  "gold_conflict_fields": ["entity"],
  "edit_type": "entity_swap"
}
```

### 4.7 正样本保留

原本 non-OOC 样本也要保留：

```json
{
  "label": 0,
  "gold_mismatch_type": "none",
  "gold_conflict_fields": [],
  "edit_type": "none"
}
```

这样模型不仅学习“错在哪里”，也学习“没有错配时不要乱解释”。

### 4.8 构造后校验

反事实样本不能替换完就直接用，必须自动校验。

校验项包括：

```text
1. target_field_changed
   目标字段确实变化。

2. replacement_type_valid
   替换词类型一致，例如 PERSON 换 PERSON。

3. edited_caption_valid
   新句子非空、长度合理、没有乱码。

4. other_fields_preserved
   非目标字段尽量没有改变。

5. duplicate_check
   edited_caption 不应大量重复。

6. group_split_check
   同一个 source_sample_id 不能同时出现在 train/val/test。
```

最重要的是 group split。否则同一原始样本的不同替换版本如果同时出现在训练集和测试集，模型会虚高。

### 4.9 输出格式

```json
{
  "sample_id": "cf_000001",
  "source_sample_id": "orig_123",
  "image_id": "img_456",
  "image_path": "...",

  "original_caption": "Biden spoke in Washington in 2024.",
  "edited_caption": "Biden spoke in Paris in 2024.",

  "label": 1,
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "edit_type": "location_swap",

  "edit_span": {
    "field": "location",
    "old": "Washington",
    "new": "Paris",
    "start": 15,
    "end": 25
  },

  "split": "train"
}
```

---

## 5. 模块 B：不依赖 true context 的特征构造

### 5.1 旧特征为什么要改

旧版 feature vector 包含：

```text
current_caption vs true_context NLI
true_context field overlap
true_context graph alignment
```

这会导致推理阶段必须输入 true_context。

新版必须改为：

```text
image + current_caption features
```

### 5.2 新版特征总览

Attribution Head 的输入特征应该来自：

```text
1. 全局图文匹配特征
2. 字段级 prompt grounding 特征
3. 字段存在性和类型特征
4. VDT score / VDT intermediate features
5. 可选视觉描述一致性特征
```

这些都不需要 true context。

---

## 6. 模块 C：全局图文匹配特征

### 6.1 这个模块做什么

用冻结视觉语言模型判断：

```text
image 与 caption 整体是否匹配
```

例如用 CLIP：

```text
image_embedding = CLIP_image_encoder(image)
text_embedding = CLIP_text_encoder(caption)
global_similarity = cosine(image_embedding, text_embedding)
```

### 6.2 输入输出

输入：

```text
image I
caption T
```

输出：

```json
{
  "clip_global_similarity": 0.31
}
```

### 6.3 它有什么用

如果整体相似度很低，说明图文可能错配。

但它不能告诉我们错在哪里，所以只是基础特征。

### 6.4 为什么不能只靠它

MUSE 相关工作提醒：相似度 shortcut 在 OOC 任务中很强，但它关注的是表面相似度，不一定判断事实矛盾。我们的 attribution head 不能只使用 global similarity，否则可能只学会“像不像”，而不是“哪里错”。

---

## 7. 模块 D：字段级 prompt grounding 特征

这是新版最关键的模块。

### 7.1 这个模块解决什么问题

我们需要知道 caption 中每个字段是否被图片支持。

caption：

```text
Biden spoke in Paris in 2024.
```

字段：

```text
entity: Biden
location: Paris
time: 2024
event_type: politics
relation: spoke
```

我们希望分别问图片：

```text
图片像不像“Biden”？
图片像不像“Paris”？
图片像不像“2024”？
图片像不像“political speech”？
```

这就是 field-aware grounding。

### 7.2 如何构造 prompt

对每个字段构造自然语言 prompt。

#### entity prompt

```text
a photo involving {entity}
a news photo of {entity}
a photo showing {entity}
```

例：

```text
a photo involving Biden
```

#### location prompt

```text
a photo taken in {location}
a news photo from {location}
a scene in {location}
```

例：

```text
a photo taken in Paris
```

#### time prompt

```text
a news photo from {time}
a photo taken in {time}
```

例：

```text
a news photo from 2024
```

注意：时间字段从图片本身很难判断，所以 time prompt 可信度低，后面必须允许 uncertain。

#### event_type prompt

```text
a photo of a {event_type}
a news photo about {event_type}
```

例：

```text
a photo of a protest
```

#### relation prompt

```text
a photo of people {relation}
a photo showing {relation}
```

例：

```text
a photo of people protesting
```

### 7.3 如何计算 grounding score

用冻结 CLIP 或类似模型：

```text
s_field = cosine(CLIP_image(image), CLIP_text(prompt_field))
```

例如：

```json
{
  "entity_prompt_sim": 0.42,
  "location_prompt_sim": 0.18,
  "time_prompt_sim": 0.05,
  "event_type_prompt_sim": 0.51,
  "relation_prompt_sim": 0.48
}
```

### 7.4 如何解释这些分数

如果 caption 是：

```text
Biden spoke in Paris in 2024.
```

图像实际是 Washington 演讲现场，可能出现：

```text
entity_prompt_sim(Biden) 高
event_type_prompt_sim(political speech) 高
location_prompt_sim(Paris) 低
```

Attribution Head 学到：

```text
entity 支持、事件类型支持，但地点不支持
=> location mismatch
```

### 7.5 它为什么不依赖 true context

因为所有分数都是从：

```text
image + current_caption fields
```

得到的，不需要原始 caption 或 true context。

### 7.6 局限

图片不一定包含地点或时间证据。

例如普通街景很难判断是 Paris 还是 London；普通新闻图很难判断年份。因此系统要允许：

```text
uncertain / insufficient visual evidence
```

---

## 8. 模块 E：字段存在性与编辑类型无关特征

为了让模型知道当前 caption 中包含哪些字段，需要加入字段存在性特征：

```text
has_entity
has_location
has_time
has_event_type
has_relation
num_entities
num_locations
num_times
```

例如：

```json
{
  "has_entity": 1,
  "has_location": 1,
  "has_time": 1,
  "num_entities": 1,
  "num_locations": 1,
  "num_times": 1
}
```

作用：

```text
如果 caption 中没有时间字段，就不应该输出 temporal mismatch。
```

---

## 9. 模块 F：VDT 分数和中间特征

### 9.1 VDT score

VDT 输出：

```json
{
  "vdt_label": "OOC",
  "vdt_score": 0.87
}
```

Attribution Head 可以使用：

```text
vdt_score
vdt_label_onehot
```

作用：

```text
VDT score 高，说明图文整体不一致概率高。
```

### 9.2 VDT embedding，可选

如果能从 VDT 取中间层 multimodal embedding，可以加入：

```text
vdt_multimodal_embedding
```

但第一版可以先不用，避免工程复杂。

---

## 10. 模块 G：视觉描述一致性特征（可选）

### 10.1 为什么需要

某些字段 CLIP prompt 不够稳定，可以让图像 captioning 模型先生成视觉描述：

```text
image -> visual_caption
```

例如：

```text
A man speaking at a podium in front of flags.
```

然后比较：

```text
current_caption vs visual_caption
```

注意：visual_caption 不是 true context。它是从图片像素生成的视觉描述，不依赖 VisualNews metadata。

### 10.2 如何使用

从 visual_caption 抽事件字段：

```text
visual_caption_event
```

从 current_caption 抽事件字段：

```text
text_event
```

计算：

```text
entity_overlap
location_overlap
event_type_overlap
relation_overlap
```

### 10.3 局限

图像 captioning 模型通常不可靠识别具体人物、地点和时间。因此它只能作为辅助特征。

---

## 11. 模块 H：Attribution Head

### 11.1 它是什么

Attribution Head 是一个轻量监督分类器。

它不是 VDT，不是 CLIP，不是大模型。它学习的是：

```text
image-caption 多模态特征模式 -> 错配类型
```

### 11.2 输入

新版输入不包含 true context。输入向量可以写成：

```text
x = [
  global_image_text_similarity,
  entity_prompt_sim,
  location_prompt_sim,
  time_prompt_sim,
  event_type_prompt_sim,
  relation_prompt_sim,
  field_presence_features,
  VDT score,
  optional visual_caption_consistency_features
]
```

### 11.3 输出

两个输出头：

#### mismatch_type head

```text
none
location mismatch
temporal mismatch
entity mismatch
uncertain
```

第一版建议只做：

```text
none
location mismatch
temporal mismatch
entity mismatch
```

#### conflict_fields head

```text
entity
location
time
```

### 11.4 模型结构

第一版可以是 MLP：

```text
input features
  -> Linear
  -> ReLU
  -> Dropout
  -> Linear
  -> mismatch_type logits
```

多标签字段头：

```text
input features
  -> Linear
  -> ReLU
  -> Dropout
  -> Linear
  -> conflict_fields logits
```

### 11.5 损失函数

```text
L = CE(mismatch_type) + λ * BCE(conflict_fields)
```

其中：

```text
CE: 多分类交叉熵
BCE: 多标签二元交叉熵
λ: 默认 1.0
```

### 11.6 为什么它能泛化

因为它不是记住某条 caption，而是学习规律：

```text
当某类字段的视觉 grounding 分数异常低，而其他字段相对正常时，通常对应这个字段的错配。
```

例如：

```text
location prompt similarity 低
entity prompt similarity 高
event_type prompt similarity 高
=> location mismatch
```

---

## 12. 数据集外样本为什么可以输出错配类型

### 12.1 前提

新版推理只需要：

```text
image + current_caption
```

不需要：

```text
true_image_context
```

因此对数据集外图片也可以运行。

### 12.2 推理流程

```text
输入外部 image + caption
        ↓
VDT 或其他 OOC detector 判断是否 OOC
        ↓
抽取 caption 字段
        ↓
构造 field prompts
        ↓
计算 image 与每个 prompt 的 grounding score
        ↓
Attribution Head 输出 mismatch_type
```

### 12.3 必须诚实说明的限制

如果图片本身无法提供足够证据，例如：

```text
时间字段：图片看不出是 2020 还是 2024
地点字段：普通街景看不出是 Paris 还是 London
实体字段：人脸不可见或模型不认识人物
```

系统应该输出：

```text
uncertain / insufficient visual evidence
```

而不是强行解释。

---

## 13. 训练集、验证集、测试集划分

### 13.1 group split

必须按 `source_sample_id` 划分。

错误做法：

```text
同一个原始样本的 location_swap 在 train
time_swap 在 test
```

这会导致数据泄漏。

正确做法：

```text
同一个 source_sample_id 产生的所有变体都进入同一个 split。
```

### 13.2 推荐比例

```text
train: 70%
val: 15%
test: 15%
```

### 13.3 泄漏检查

检查项：

```text
source_sample_id 跨 split 数量 = 0
image_id 跨 split 数量 = 0 或单独报告
edited_caption 重复数量
类别分布
```

---

## 14. 实验设计

### 14.1 反事实生成质量

| Edit type | Generated | Kept | Keep rate |
|---|---:|---:|---:|
| location_swap | 待跑 | 待跑 | 待跑 |
| time_swap | 待跑 | 待跑 | 待跑 |
| entity_swap | 待跑 | 待跑 | 待跑 |

### 14.2 Synthetic held-out test

模型在反事实测试集上的结果：

| Method | Uses true context at inference? | Type Acc | Field Micro-F1 | Exact Match |
|---|---|---:|---:|---:|
| Majority | No | 待跑 | 待跑 | 待跑 |
| Global CLIP similarity | No | 待跑 | 待跑 | 待跑 |
| Field prompt grounding | No | 待跑 | 待跑 | 待跑 |
| Attribution Head | No | 待跑 | 待跑 | 待跑 |
| COVE-lite oracle | Yes | 待跑 | 待跑 | 待跑 |

注意：COVE-lite oracle 可以做，但要单独标注 `Uses true context = Yes`，不能和最终模型混在一起。

### 14.3 学习曲线

| MaxPerType | Train size | Type Acc | Field Micro-F1 | Exact Match |
|---:|---:|---:|---:|---:|
| 80 | 待跑 | 待跑 | 待跑 | 待跑 |
| 200 | 待跑 | 待跑 | 待跑 | 待跑 |
| 1000 | 待跑 | 待跑 | 待跑 | 待跑 |
| 3000 | 待跑 | 待跑 | 待跑 | 待跑 |

### 14.4 消融实验

| Variant | 目的 |
|---|---|
| w/o field prompt grounding | 验证字段级视觉对齐是否有效 |
| w/o VDT score | 验证 VDT 分数是否有帮助 |
| w/o field presence | 验证字段存在性特征是否有帮助 |
| only global similarity | 验证相似度 shortcut baseline |
| full attribution head | 完整模型 |

### 14.5 真实 OOC 人工评估

构建人工评测集：

```text
examples/real_ooc_attribution_eval_set.jsonl
```

标注字段：

```text
gold_mismatch_type
gold_conflict_fields
rationale
```

推理时不输入 true context；标注时可以参考 true context。

---

## 15. 当前旧实验结果如何重新解释

你们之前跑出的：

```text
attribution head MLP Type Acc = 0.9268
Field Micro-F1 = 0.9474
```

如果它使用了 `current_caption vs true_context` 的 NLI/evidence/graph 特征，那么它应被重新命名为：

```text
COVE-lite oracle-feature Attribution Head
```

它不能作为最终 no-true-context 模型结果。

正确写法：

```text
在 oracle-feature 设置下，使用 true context 相关特征的 attribution head 在 synthetic test 上取得较高结果；这说明反事实标签可学习。但最终系统需要进一步替换为不依赖 true context 的 image-caption feature head。
```

也就是说，这个结果不是废掉，而是变成：

```text
oracle upper-bound / feasibility proof
```

---

## 16. Codex 需要实现的新脚本

### 16.1 图文特征构造

```text
scripts/features/build_image_caption_attribution_features.py
```

输入：

```text
counterfactual_attribution_train.jsonl
counterfactual_attribution_val.jsonl
counterfactual_attribution_test.jsonl
image root / image path mapping
```

禁止使用：

```text
true_image_context
true_context_event
current_caption vs true_context NLI
```

允许使用：

```text
image embedding
text embedding
CLIP image-text similarity
field prompt image similarity
VDT score
visual caption features
field presence features
```

输出：

```text
attribution_features_train.csv
attribution_features_val.csv
attribution_features_test.csv
```

### 16.2 训练脚本修改

现有：

```text
scripts/train/train_attribution_head.py
```

需要支持读取 CSV 特征，而不是只从 JSONL 中读 true_context 相关特征。

### 16.3 推理脚本

```text
scripts/infer/infer_vdt_cf_attr.py
```

输入：

```text
--image path/to/image.jpg
--caption "..."
```

输出：

```json
{
  "vdt_label": "OOC",
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "confidence": 0.82,
  "uses_true_context": false
}
```

---

## 17. 答辩讲稿

可以这样讲：

> 我们一开始考虑利用 VisualNews 的原始上下文作为 true context，直接比较当前 caption 和 true context 来解释错配原因。但这种方式的问题是，推理阶段仍然依赖图片原始上下文；如果输入一张数据集外图片，系统无法保证拿到 true context。因此我们进一步提出基于可控反事实训练的 VDT-CF-Attr。训练阶段，我们只从 non-OOC 匹配样本出发，对 caption 中的实体、地点或时间做单字段最小编辑，生成带有确定错配原因的 OOC 反事实样本。推理阶段，Attribution Head 不再输入 true context，而是基于 image-caption 的多模态特征、字段级 prompt grounding 特征和 VDT score 输出错配类型。这样系统可以在没有原始上下文的外部样本上进行错配原因预测；如果图片本身无法提供足够视觉证据，则输出 uncertain。

---

## 18. 最终边界

必须明确：

```text
我们不是说模型能在任何情况下都准确判断错配原因。
```

它能判断的前提是：

```text
1. VDT 或主分类器已经判断图文可能错配；
2. caption 中有可抽取字段；
3. 图片中有足够视觉证据支持或反驳这些字段；
4. 模型训练中见过类似字段冲突模式。
```

对于时间、地点这类不一定能从图片直接看出的字段，必须允许：

```text
uncertain / insufficient visual evidence
```

---

## 19. 最终一句话总结

```text
VDT-CF-Attr 的核心不是用 true context 直接比较文本，而是利用 true context 和 non-OOC 样本在训练阶段构造可控反事实监督信号，训练一个推理阶段只依赖 image + current_caption 的 attribution head，从而在 VDT 判断 OOC 后输出可能的错配原因。
```

---

## 20. 参考文献方向

1. NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media.
2. VDT: Out-of-Context Misinformation Detection via Variational Domain-Invariant Learning with Test-Time Training.
3. COVE: COntext and VEracity prediction for out-of-context images.
4. Counterfactually Augmented Data: Learning the Difference that Makes a Difference with Counterfactually-Augmented Data.
5. Similarity over Factuality / MUSE: Are we making progress on multimodal out-of-context misinformation detection?
