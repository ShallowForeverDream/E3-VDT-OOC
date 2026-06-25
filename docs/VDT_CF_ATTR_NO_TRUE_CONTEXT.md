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
  -> VDT 判断 OOC / Non-OOC
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
VDT score / VDT label
```

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

## 4. 当前本地结果

配置：

```text
MaxPerType = 80
group split = source_sample_id / image_id / text_id
true context at inference = False
CLIP = openai/clip-vit-base-patch32
test = 38
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
| majority | False | 0.2368 | 0.2812 | 0.2368 |
| field prompt grounding rule | False | 0.3421 | 0.3415 | 0.3947 |
| logistic regression no-true-context | False | 0.4474 | 0.3939 | 0.2632 |
| image+caption MLP attribution head | False | 0.4474 | 0.3500 | 0.4474 |

## 5. 如何解释这个结果

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

## 6. 最稳答辩口径

> 我们最初实现了 COVE-lite oracle attribution，用于说明“有真实上下文时可以定位字段冲突”。但这个设置推理阶段依赖 true context，实际应用受限。因此我们进一步实现 VDT-CF-Attr：用 Non-OOC 图文对构造单字段反事实错配，训练阶段得到干净错配类型标签；推理阶段 attribution head 只接收 image、current caption 和 VDT 输出，不读取 true context。当前 no-true-context 结果仍低于 oracle，说明仅靠图像判断人物、地点、时间错配更难，后续需要 OCR/检索/更强 VLM caption 作为视觉证据增强。

