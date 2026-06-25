# 期末报告初稿：VDT-COVE-Attr 跨域图文内容挪用检测与错配归因系统

> 用途：报告负责人按课程模板排版。本文档为当前最终研究口径，不再使用“E3-VDT Full 显著提升分类指标”的旧表述。

## 摘要

新闻场景中的 Out-of-Context（OOC）图文内容挪用通常不修改图像像素，而是将真实图片放入错误文本语境中，造成误导性叙事。针对该问题，本项目首先复现 VDT 作为跨域 OOC 二分类 baseline，在 NewsCLIPpings / VisualNews strict BLIP-2/GaussianBlur 设置下完成两组核心实验：`target_domain=bbc,guardian` 得到 `F1=0.7353`、`Acc=0.7383`、`AUC=0.7398`；`target_domain=usa_today,washington_post` 在 RTX 4060 Laptop 显存约束下降 batch size 至 64，得到 `F1=0.8032`、`Acc=0.8032`、`AUC=0.8028`。

VDT 能判断图文对是否 OOC，但不输出错配原因。为弥补这一不足，本项目提出 **VDT-COVE-Attr**：在不覆盖 VDT 主分类结果的前提下，借鉴 COVE 的 context-first 思路，用 VisualNews 原始上下文构造图像真实语境（true image context），再比较当前 caption 与真实上下文中的 `entity/location/time/event_type/relation` 五类事件字段，输出 `mismatch_type`、`conflict_fields`、`event_scores` 和结构化解释。为避免将规则输出直接包装成真值，本项目进一步设计人工归因评测协议，以 `mismatch_type_accuracy`、`conflict_field_micro_f1`、`macro_f1` 和 `exact_match_rate` 验证解释是否优于 majority、sampled、text-only 等简单 baseline。

本项目的贡献不是提出新的主分类 SOTA，而是完成 VDT baseline 复现，并将其二分类输出扩展为可解释、可评测、可展示的内容安全审核系统。

## 1. 引言

OOC 图文挪用的风险在于：图像本身真实，文本也可能局部合理，但二者组合后指向错误事件。传统图像篡改检测难以发现这类问题；普通图文相似度也可能被同主题不同事件误导。因此，内容安全系统不仅需要判断是否 OOC，还需要回答“错在哪里、依据是什么”。

本项目以 VDT 为主 baseline，复现其跨域 OOC 检测流程；随后在 VDT 之外增加 context-grounded attribution 模块，面向人工审核输出可解释字段。

## 2. 相关工作

### 2.1 VDT：跨域 OOC 检测 baseline

VDT 通过变分域不变表示和测试时训练增强 NewsCLIPpings 中跨新闻机构域的泛化能力。它适合作为 OOC 二分类 baseline，但不输出细粒度错配字段。因此本项目将 VDT 用作主分类模块，而不是把解释模块混入 VDT 主分类。

### 2.2 SNIFFER：解释需要外部上下文和证据

SNIFFER 使用多模态大模型、指令微调和外部检索证据进行 explainable OOC detection。它说明 OOC 解释不能只依赖分类分数，而需要上下文或证据支撑。受限于课程项目算力，本项目不完整复现 SNIFFER，而是吸收其“contextual verification”思想。

### 2.3 COVE：先恢复图像真实上下文，再判断 caption

COVE 将 OOC 验证拆成 true context prediction 与 caption veracity prediction。该思路最适合本项目适配：在 NewsCLIPpings / VisualNews 中，我们可以用 VisualNews 原始 caption/title/article metadata 作为 true image context，形成 COVE-lite。

### 2.4 MUSE：similarity shortcut 警告

MUSE 表明简单相似度特征可能在 OOC benchmark 上取得很强表现，因此不能只用总体 Accuracy/F1 证明模型真正理解事实一致性。本项目通过 hard negative 与人工归因评测检验解释模块是否只是相似度 shortcut。

### 2.5 AMG 与归因评测

多模态假新闻数据集通常只有二分类标签，缺少细粒度 attribution labels。AMG 类工作说明，若要声称解释有效，必须构造归因标签和评测协议。因此本项目新增人工 attribution set，用字段级 F1 验证解释可靠性。

## 3. 方法：VDT-COVE-Attr

### 3.1 总体结构

```text
Input image-caption pair
        |
        v
VDT baseline -> OOC / Non-OOC + confidence
        |
        v
COVE-lite true context recovery
        |
        v
Event-field attribution sidecar
        |
        v
mismatch_type + conflict_fields + event_scores + explanation
```

### 3.2 Accuracy-preserving 主分类策略

正式策略中：

```text
final_label = vdt_label
final_score = vdt_score
```

解释模块不覆盖 VDT 主分类，因此分类 Accuracy/F1 与 VDT baseline 持平。只有后续验证集证明融合事件分数不降低指标，才可尝试 guarded fusion；当前不把 fusion 作为已完成贡献。

### 3.3 COVE-lite 图像真实上下文构造

对每个 NewsCLIPpings 样本，根据 `image_id` 在 VisualNews metadata 中寻找图像原始上下文：

```text
image_id -> VisualNews original caption/title/article metadata -> true_image_context
```

构造输出：

```json
{
  "sample_id": "...",
  "image_id": "...",
  "current_caption": "...",
  "true_image_context": "...",
  "label": 1,
  "domain": "bbc"
}
```

这一步将解释从“手填图片描述”升级为“基于图像真实上下文的证据 grounding”。

### 3.4 事件字段抽取与归因

分别从 current caption 和 true image context 中抽取：

```text
entity / location / time / event_type / relation
```

计算字段一致性：

```text
C_event = [s_entity, s_location, s_time, s_event_type, s_relation]
```

若某字段一致性低于阈值，则输出为 `conflict_fields`。主错配类型由冲突字段映射得到：

| 字段 | 错配类型 |
|---|---|
| entity | entity mismatch |
| location | location mismatch |
| time | temporal mismatch |
| event_type | event-type mismatch |
| relation | relation mismatch |

### 3.5 RED-DOT-lite 证据相关性思想

外部上下文也可能缺失或无关。本项目的轻量处理是：若 true image context 缺失、字段抽取不足或关键字段均无法比较，则输出 `uncertain / evidence insufficient`，而不是强行生成解释。后续可以加入更完整的 evidence relevance scoring。

## 4. 实验设计

### 4.1 VDT baseline 复现

| 设置 | 状态 | F1 | Acc | AUC | 说明 |
|---|---|---:|---:|---:|---|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 | bs256/CUBLAS 不稳定后降到 bs128 |
| `usa_today,washington_post`, bs128 | failed_oom | - | - | - | 显存不足，保留失败日志 |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 | 本机可复现设置 |

准确表述：完成 VDT strict BLIP-2/GaussianBlur 核心设置下两组跨域 baseline 复现，不夸大为完整复现论文全部实验。

### 4.2 COVE-lite coverage 实验

运行脚本：

```powershell
.\scripts\run_cove_lite_attribution_experiments.ps1 `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -MaxRecords 500 `
  -EvalSampleN 50
```

报告：

```text
total samples
kept samples
coverage = kept / total
missing_text
missing_true_context
```

### 4.3 人工归因评测

人工标注文件：

```text
examples/attribution_eval_set.jsonl
```

每条标注：

```json
{
  "gold_mismatch_type": "location mismatch",
  "gold_conflict_fields": ["location"],
  "rationale": "...",
  "annotation_status": "done"
}
```

### 4.4 Baseline 对比

| 方法 | 说明 |
|---|---|
| majority | 永远预测最常见类型 |
| sampled | 随机采样类型 |
| text-only | 不看 true context，只输出证据不足 |
| COVE-lite event rule | current caption vs true context 字段比较 |

指标：

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

只有当 COVE-lite event rule 超过简单 baseline，才在报告中写成解释增强有效；否则必须写失败分析。

## 5. 系统实现

仓库实现：

- `demo/app.py`：Gradio 展示系统；
- `src/e3vdt/inference/pipeline.py`：统一推理管线；
- `scripts/context/build_cove_lite_context_pairs.py`：COVE-lite 上下文构造；
- `scripts/labels/build_weak_attribution_from_context.py`：弱归因标签；
- `scripts/eval/build_attribution_eval_sample.py`：人工标注候选抽样；
- `scripts/eval/run_attribution_baselines.py`：归因 baseline 评测；
- `scripts/run_cove_lite_attribution_experiments.ps1`：本地一键实验脚本。

## 6. 结果与讨论

当前可以确定的结果：

1. VDT 两组核心复现已完成；
2. VDT-COVE-Attr 系统结构已实现；
3. 分类策略为 baseline-preserving，因此主分类指标不低于 VDT baseline；
4. 解释模块的有效性必须等待人工归因评测结果支撑。

因此，本项目现阶段不写“分类性能显著提升”，而写“在保持 VDT 分类结果的前提下增加 context-grounded attribution，并设计可验证实验协议”。

## 7. 不足与后续工作

1. COVE-lite 依赖 VisualNews metadata，coverage 需要实际统计；
2. 事件字段抽取仍是轻量规则，后续可接入更强 NER、OCR、captioning 或 LLM；
3. 人工归因集规模有限，需要报告样本量限制；
4. Evidence Gate、Event-Guided TTT 和 attribution head 训练属于扩展方向，当前不作为已完成主贡献。

## 8. 成员分工

| 角色 | 贡献 |
|---|---|
| 组长 | 路线设计、VDT 复现、系统集成、GitHub 维护、答辩主线 |
| 复现负责人 | 数据准备、BLIP-2/GaussianBlur 预处理、VDT baseline 训练、指标整理 |
| 创新模块负责人 | COVE-lite 上下文构造、归因标签、人工评测集 |
| 系统/报告负责人 | Gradio demo、报告/PPT、案例分析 |

## 9. 最终答辩口径

> 我们完成了 VDT strict baseline 的核心复现。VDT 负责 OOC / Non-OOC 主分类。针对 VDT 不解释错配原因的问题，我们借鉴 COVE 的 context-first 思路，用 VisualNews 原始上下文构造图像真实语境，再比较 current caption 与 true context 的事件字段差异，输出 mismatch type 和 conflict fields。解释模块不覆盖 VDT 主分类，因此分类指标保持 baseline。解释是否有效通过人工归因集上的 field-F1 与 type accuracy 验证。
