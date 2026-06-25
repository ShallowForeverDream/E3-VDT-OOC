# VDT-COVE-Attr v2 技术路线

## 1. 任务定义

本项目研究新闻图文内容挪用检测，即真实图片被错误文本重新配对后造成误导传播的问题。输入为 `image + current caption`，主分类任务输出 `OOC / Non-OOC`；可解释任务输出 `mismatch_type / conflict_fields / evidence_relevance / explanation`。

本项目不做图像篡改检测、deepfake 检测或纯文本谣言检测。图片本身可以是真实的；问题在于图片被置于错误语境中。

## 2. 数据集

- **NewsCLIPpings**：主 OOC 二分类数据集，提供图文是否匹配标签。
- **VisualNews metadata**：通过 `image_id` 获得图片原始新闻上下文，作为 `true_image_context`。
- **人工 attribution eval set**：我们后续标注的小规模归因评测集，提供 `gold_mismatch_type` 与 `gold_conflict_fields`。

## 3. Baseline

主 baseline 是 VDT。VDT 解决跨域 OOC 二分类问题，输出 `OOC / Non-OOC`。我们已经完成两组 strict BLIP-2/GaussianBlur 复现：

| Setting | Batch | F1 | Acc | AUC |
|---|---:|---:|---:|---:|
| target=`bbc,guardian` | 128 | 0.7353 | 0.7383 | 0.7398 |
| target=`usa_today,washington_post` | 64 | 0.8032 | 0.8032 | 0.8028 |

VDT 的不足是：只告诉我们是否 OOC，不能解释错配字段。

## 4. 技术路线

VDT-COVE-Attr v2 采用 sidecar 方式，不覆盖 VDT 主分类。

```text
image + caption
    ↓
VDT baseline → OOC / Non-OOC + confidence
    ↓
COVE-lite true context construction
    image_id → VisualNews original caption/title/metadata
    ↓
Structured event extraction
    current_caption → current_event_tuple
    true_context → true_event_tuple
    ↓
Evidence relevance / sufficiency scoring
    ↓
Field-wise NLI contradiction detection
    entity / location / time / event_type / relation
    ↓
Attribution output
    mismatch_type + conflict_fields + explanation
```

## 5. 两个核心问题

### 问题 A：如何从文本抽取事件标签？

高标准方案不是纯正则，而是分层抽取：

1. rule extractor：作为 baseline。
2. NER/date/location/OpenIE/SRL：作为可复现增强。
3. LLM JSON extractor：可选增强，必须缓存并人工评估。

输出 event tuple：

```json
{
  "entities": [],
  "locations": [],
  "times": [],
  "event_types": [],
  "relations": []
}
```

### 问题 B：如何判断图片被如何误用？

比较 `current_caption event tuple` 与 `true_image_context event tuple`。核心不是字符串相似度，而是 field-wise NLI / claim-evidence contradiction：

- current caption 是否被 true image context 支持？
- 哪些字段 contradiction？
- true context 是否足够相关和充分？

输出：

```json
{
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "field_nli": {
    "location": "contradiction",
    "time": "neutral"
  }
}
```

## 6. 创新点

1. **VDT 可解释扩展**：保留 VDT 主分类，增加可验证的 attribution sidecar。
2. **COVE-lite true context**：利用 VisualNews 原始上下文构造图片真实语境。
3. **Field-wise NLI 归因**：逐字段判断当前语境与真实语境的 entailment / contradiction / neutral。
4. **Evidence relevance / sufficiency**：避免上下文不足时强行解释。
5. **AMG-style 人工归因评测**：用人工 gold set 评估 mismatch type accuracy 与 conflict field F1。

## 7. 不能过度声称

不要声称：

- 我们超过 VDT 主分类性能。
- NewsCLIPpings 自带归因标签。
- 弱标签是真实标签。
- 规则模块是最终创新。

应该声称：

- 我们复现 VDT，并针对其缺少解释的问题提出上下文归因扩展。
- 归因效果通过人工子集与 baseline 对比验证。
