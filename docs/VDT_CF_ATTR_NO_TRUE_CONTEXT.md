# VDT-CF-Attr：不依赖 true context 的图文错配归因路线

这份文档记录最新技术路线修正：

> `true_image_context` 只能用于构造训练样本和人工评测参考，不能作为系统推理阶段的必需输入。

## 1. 为什么要改

上一版 `VDT-COVE-Attr` 的主逻辑是：

```text
current_caption + true_image_context -> 字段冲突 -> mismatch_type
```

这适合做 oracle / COVE-lite baseline，但推理阶段依赖 VisualNews 原始上下文。对数据集外图片，这个输入通常不存在。

新的最终系统应是：

```text
image + current_caption
  -> VDTAdapter 自动判断 OOC / Non-OOC / Uncertain
  -> Attribution Head 判断错配类型
```

因此我们新增 `VDT-CF-Attr` 路线：用 controlled counterfactual 构造监督标签，但训练和推理的 attribution head 不读取 `true_image_context`。

## 2. 训练和推理定义

### 训练阶段

从 Non-OOC 匹配样本构造：

```text
image I + original_caption T -> label=Non-OOC
image I + edited_caption T' -> label=OOC, mismatch_type=被编辑字段
```

例如：

```text
T : Biden spoke in Washington in 2024.
T': Biden spoke in Paris in 2024.
gold_mismatch_type = location mismatch
```

### 推理阶段

只允许输入：

```text
image
current_caption
```

系统内部自动产生：

```text
VDT score / VDT label
```

当前实现位于 `src/e3vdt/inference/vdt_adapter.py`：本机有 no-true-context 实验特征表时优先训练轻量二分类 head；没有特征表时回退到 CLIP image-caption similarity；COVE/oracle 页面没有原图输入时，用 current caption 与 true context 的事件一致性产生 VDT-compatible label。它的作用是让 demo 不再手填 `VDT label / score`，但不能包装成“官方 BLIP-2 VDT checkpoint 已完整在线接入”。

不允许输入：

```text
true_image_context
VisualNews original caption
current_caption vs true_context NLI
true_context graph alignment
```

## 3. 当前已实现脚本

### 构建 image+caption 特征

```text
scripts/features/build_image_caption_attribution_features.py
```

输入：

```text
controlled_counterfactual_train/val/test.jsonl
```

输出：

```text
image_caption_features_train.csv
image_caption_features_val.csv
image_caption_features_test.csv
```

特征包括：

```text
CLIP image-caption similarity
CLIP image-field-prompt similarity
caption field presence/count
caption length/token count
VDT score placeholder or provided score
```

其中字段 prompt 例子：

```text
entity   -> a news photo involving Biden
location -> a news photo taken in Paris
time     -> a news photo from 2024
event    -> a news photo about protest
relation -> a news photo showing speak
```

脚本会从 `origin/data.json` 找 `image_id -> image_path`，再从 `origin.tar` 按 tar index 直接读取图片，不需要解压 97GB 原图。

### 训练 no-true-context attribution head

```text
scripts/train/train_no_true_context_attribution_head.py
```

输入是 CSV 特征，不读取 true context。

### 一键实验

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_attr_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 80 `
  -OutputDir outputs\no_true_context_attr `
  -Device cuda `
  -BatchSize 16
```

快速无 CLIP smoke：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_attr_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 20 `
  -OutputDir outputs\no_true_context_attr_smoke `
  -NoClip `
  -Device cpu
```

### 80/200/1000 扩展实验

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_scaling.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -Sizes "80,200,1000" `
  -ContextPairs outputs\cove_lite_context_pairs_3000.jsonl `
  -Device cuda `
  -BatchSize 24
```

如果 `outputs\cove_lite_context_pairs_3000.jsonl` 不存在，可以让脚本现场构造：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_scaling.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -Sizes "80,200,1000" `
  -ContextPairs outputs\cove_lite_context_pairs_3000.jsonl `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -ContextMaxRecords 3000 `
  -RebuildContextPairs `
  -Device cuda `
  -BatchSize 24
```

## 4. 当前本地结果

配置：

```text
MaxPerType = 80
group split = source_sample_id / image_id / text_id
true context at inference = False
CLIP = openai/clip-vit-base-patch32
test = 42
class counts = none/location/time/entity = 80/80/80/80
```

泄漏检查：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross-split duplicate caption = 0
```

结果：

| Method | Uses true context at inference? | Type Acc | Field Micro-F1 | Exact Match |
|---|---|---:|---:|---:|
| majority | False | 0.1429 | 0.1690 | 0.1429 |
| field prompt grounding rule | False | 0.2857 | 0.2381 | 0.3333 |
| logistic regression no-true-context | False | 0.4286 | 0.5301 | 0.2619 |
| image+caption MLP attribution head | False | 0.3571 | 0.3667 | 0.2619 |

补充：已修复 `time_swap` 不平衡问题。当前 `MaxPerType=80` 下 `time_swap=80`，不再是旧版的 27。

### Scaling 结果

本地使用 `outputs\cove_lite_context_pairs_3000.jsonl` 跑完 `MaxPerType=80/200/1000`。三组泄漏检查均为 0。

| MaxPerType | Counts none/location/time/entity | Best stable method | Type Acc | Field Micro-F1 | Exact Match |
|---:|---|---|---:|---:|---:|
| 80 | 80/80/80/80 | logistic regression no-true-context | 0.2745 | 0.3564 | 0.1961 |
| 200 | 200/200/200/200 | logistic regression no-true-context | 0.4266 | 0.5195 | 0.2308 |
| 1000 | 1000/797/1000/1000 | logistic regression no-true-context | 0.5275 | 0.5719 | 0.3250 |

解释：`MaxPerType=1000` 时 location 只有 797，是因为 `cove_lite_context_pairs_3000` 里可抽取地点 span 的 Non-OOC caption 不足；time 类通过 `YYYY -> YYYY` 多反事实变体补到 1000。学习曲线显示，no-true-context 设置下 logistic regression head 比 prompt rule 更稳定；MLP 还没有稳定超过 LR，因此报告里不要把 MLP 包装成最终最优模型。

## 5. 推理后处理约束

已在 `scripts/infer/infer_vdt_cf_attr.py` 增加 field-presence constraint：

```text
如果 caption 中 entity_present=0，则最终 JSON 不能输出 entity mismatch；
如果 location/time/event_type/relation 不存在，同理不能输出对应 mismatch；
无有效字段时，OOC 输出 uncertain / insufficient visual evidence，Non-OOC 输出 benign illustrative image。
```

JSON 会额外输出：

```json
{
  "postprocess_applied": true,
  "postprocess_reason": "field_absent_constraint_reselected_from_present_fields"
}
```

## 6. 如何解释这个结果

这个结果比 COVE-lite oracle attribution 低，是正常且重要的：

- COVE-lite oracle 使用 `true_image_context`，信息更强；
- no-true-context 只看图片和当前 caption，更接近真实应用；
- 具体人物、地点、年份很多时候不能仅从图像可靠判断；
- time mismatch 尤其依赖 OCR、新闻水印、元数据或外部检索。

所以答辩时不能把两个表混在一起：

| 方法 | 推理时是否用 true context | 作用 |
|---|---|---|
| COVE-lite oracle / field-wise NLI | Yes | 归因上限、训练构造、人工评测辅助 |
| VDT-CF-Attr image+caption head | No | 最终实际推理路线 |

## 7. 最稳答辩口径

> 我们最初实现了 COVE-lite oracle attribution，用于说明“有真实上下文时可以定位字段冲突”。但这个设置推理阶段依赖 true context，实际应用受限。因此我们进一步实现 VDT-CF-Attr：用 Non-OOC 图文对构造单字段反事实错配，训练阶段得到干净错配类型标签；推理阶段 attribution head 只接收 image、current caption 和 VDT 输出，不读取 true context。当前 no-true-context 结果仍低于 oracle，说明仅靠图像判断人物、地点、时间错配更难，后续需要 OCR/检索/更强 VLM caption 作为视觉证据增强。

