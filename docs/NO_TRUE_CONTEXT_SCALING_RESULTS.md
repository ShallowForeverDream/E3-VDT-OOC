# No-true-context VDT-CF-Attr 修复与 scaling 结果

更新时间：2026-06-26

## 本轮完成内容

1. **推理后处理修复**
   - 文件：`scripts/infer/infer_vdt_cf_attr.py`
   - 增加 field-presence constraint：caption 中不存在的字段不能作为最终错配字段输出。
   - JSON 新增 `postprocess_applied`、`postprocess_reason`。
   - Demo 已同步显示后处理状态。
   - 单测：`tests/test_vdt_cf_attr_infer_postprocess.py`。

2. **time_swap 数据不平衡修复**
   - 文件：`scripts/data/build_controlled_counterfactuals.py`
   - 只做 `YYYY -> YYYY` 年份替换。
   - replacement pool 固定包含 `2016–2026`。
   - 允许同一 source row 产生多个 time 反事实变体；group split 会把同源变体放在同一 split，避免泄漏。
   - `counterfactual_generation_stats.json` 新增 `time_swap_summary`。

3. **no-true-context scaling 脚本**
   - 文件：`scripts/run_no_true_context_scaling.ps1`
   - 支持一次跑 `80/200/1000`。
   - 输出：`outputs/no_true_context_scaling_results.csv`。

## 验证

```text
python -m pytest tests -q
11 passed
```

推理后处理 smoke case：

```text
caption: A large protest erupted in Paris on Monday after a new climate policy.
field_presence.entity = 0
final mismatch_type != entity mismatch
postprocess_applied = true
```

## MaxPerType=80 no-true-context 本地结果

输入：`outputs/cove_lite_context_pairs.jsonl`

```text
class counts = none/location/time/entity = 80/80/80/80
leakage = 0
train/val/test = 219/59/42
```

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.1429 | 0.1690 | 0.1429 |
| field prompt grounding rule | 0.2857 | 0.2381 | 0.3333 |
| logistic regression no-true-context | 0.4286 | 0.5301 | 0.2619 |
| image+caption MLP attribution head | 0.3571 | 0.3667 | 0.2619 |

## Scaling 结果

输入：`outputs/cove_lite_context_pairs_3000.jsonl`

| MaxPerType | Counts none/location/time/entity | Test N | Logistic Type Acc | Logistic Field Micro-F1 | Logistic Exact |
|---:|---|---:|---:|---:|---:|
| 80 | 80/80/80/80 | 51 | 0.2745 | 0.3564 | 0.1961 |
| 200 | 200/200/200/200 | 143 | 0.4266 | 0.5195 | 0.2308 |
| 1000 | 1000/797/1000/1000 | 563 | 0.5275 | 0.5719 | 0.3250 |

三组 `source_sample_id / image_id / text_id / duplicate caption` 泄漏均为 0。

## 五类训练：加入 original OOC 的 different-event mismatch

为避免把所有原始 OOC 都武断标成完全不同事件，本轮只筛选满足以下条件的原始 OOC：

```text
NewsCLIPpings similarity_score <= 0.65
caption 与 true_image_context 的 token Jaccard <= 0.08
caption / true_context 至少 4 个 token
```

命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_attr_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 1000 `
  -ContextPairs outputs\cove_lite_context_pairs_3000.jsonl `
  -OutputDir outputs\no_true_context_attr_5way_1000 `
  -Device cuda `
  -BatchSize 24 `
  -IncludeDifferentEvent `
  -MaxDifferentEvent 1000 `
  -DifferentEventMaxSimilarity 0.65 `
  -DifferentEventMaxTokenJaccard 0.08
```

生成分布：

| Label | Count |
|---|---:|
| benign illustrative image / none | 1000 |
| entity mismatch | 1000 |
| location mismatch | 1000 |
| temporal mismatch | 1000 |
| different-event mismatch | 987 |

`different-event mismatch` 没到满 1000，是因为严格筛选后只剩 987 条；这里宁愿少一点，也不把相似 hard negative 错标为 different-event gold。

split 分布：

| Split | none | entity | location | time | different-event | total |
|---|---:|---:|---:|---:|---:|---:|
| train | 713 | 713 | 716 | 738 | 676 | 3556 |
| val | 154 | 154 | 152 | 120 | 148 | 728 |
| test | 133 | 133 | 132 | 142 | 163 | 703 |

泄漏检查：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross_split duplicate caption = 0
```

最佳保存模型：

```text
selected_model_name = logistic_regression_no_true_context
Type Acc = 0.4011
Field Micro-F1 = 0.5841
Exact Match = 0.3257
```

分类报告摘录：

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| benign illustrative image | 0.3396 | 0.4060 | 0.3699 | 133 |
| different-event mismatch | 0.3204 | 0.2025 | 0.2481 | 163 |
| entity mismatch | 0.3043 | 0.3158 | 0.3100 | 133 |
| location mismatch | 0.2667 | 0.1212 | 0.1667 | 132 |
| temporal mismatch | 0.5638 | 0.9648 | 0.7117 | 142 |

结论：系统现在**可以输出并训练 different-event mismatch**，但从 test 指标看，对 different-event 的区分能力还不强，不能声称已经可靠解决“完全错配 vs 单字段错配”。更稳的说法是：我们完成了五类闭环训练和初步区分，后续还需要更强视觉语义特征和人工真实 OOC 标注来提升 different-event recall。

## Plus2000：加入额外原始 OOC different-event 训练样本

用户提出的新训练设置是：

```text
none / entity / location / time ≈ 1000 / 1000 / 1000 / 1000
different-event mismatch = 先保留五类版本中的原始 OOC，再额外加入约 2000 条原始 OOC
人工标注 100 条真实 OOC gold set 不参与训练
```

本轮重新用 `outputs/cove_lite_context_pairs_10000.jsonl` 选样，筛选条件仍然保持严格：

```text
NewsCLIPpings similarity_score <= 0.65
caption 与 true_image_context 的 token Jaccard <= 0.08
caption / true_context 至少 4 个 token
排除 examples/real_ooc_attribution_eval_set.jsonl 中的 sample_id / text_id / image_id
```

命令摘录：

```powershell
python scripts\context\build_cove_lite_context_pairs.py `
  --newsclippings-data-dir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  --visualnews-metadata-dir E:\OOC_Datasets\VisualNews\articles_metadata `
  --output outputs\cove_lite_context_pairs_10000.jsonl `
  --max-records 10000 --seed 2026

powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_attr_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 1000 `
  -ContextPairs outputs\cove_lite_context_pairs_10000.jsonl `
  -OutputDir outputs\no_true_context_attr_5way_plus2000 `
  -Device cuda `
  -BatchSize 24 `
  -IncludeDifferentEvent `
  -MaxDifferentEvent 3000 `
  -DifferentEventMaxSimilarity 0.65 `
  -DifferentEventMaxTokenJaccard 0.08
```

训练数据分布：

| Label | Count |
|---|---:|
| benign illustrative image / none | 1000 |
| entity mismatch | 1000 |
| location mismatch | 1000 |
| temporal mismatch | 1000 |
| different-event mismatch | 3000 |

其中有 94 条原始 OOC 因与人工 gold set 的 `sample_id/text_id/image_id/source_sample_id` 重合而被排除。泄漏检查仍为 0：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross_split duplicate caption = 0
```

合成 held-out test 上的最佳模型变为 `attr_head_image_caption_mlp`：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.4669 | 0.7012 | 0.4669 |
| logistic regression no-true-context | 0.3317 | 0.6655 | 0.4228 |
| image+caption MLP attribution head | **0.5220** | 0.6876 | 0.3487 |

注意：由于 test 里 `different-event mismatch` 占比也更高，majority 的 field-F1 和 exact 不低；因此这张表只能说明 MLP 在五类类型准确率上优于多数类，但不能单独证明真实 OOC 泛化已经解决。

真实 OOC 人工 100 条 no-true-context 评估结果：

| Model | Type Acc | Field Micro-F1 | Exact Match | 预测分布摘要 |
|---|---:|---:|---:|---|
| `no_true_context_attr_5way_1000` | 0.0900 | 0.3276 | 0.0300 | different-event 只预测 6 条 |
| `no_true_context_attr_5way_plus2000` | **0.2900** | **0.4781** | 0.0300 | different-event 预测 33 条 |

这说明：额外加入原始 OOC 的 different-event 训练样本以后，模型在真实 OOC 上更愿意输出 `different-event mismatch`，类型准确率从 0.09 提升到 0.29，字段 micro-F1 从 0.328 提升到 0.478。结论要诚实写成“训练分布修正带来明显改善，但仍未达到可靠泛化”，不能写成已经彻底解决。

系统默认加载顺序也已更新：

```text
outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl
→ outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl
→ outputs/no_true_context_attr/no_true_context_attr_head.pkl
→ field-prompt grounding rule fallback
```

## 答辩口径

- VDT 主分类没有被修改；解释模块作为 sidecar，不覆盖 VDT 的 OOC / Non-OOC 输出。
- COVE-lite true-context 是 oracle / 上限 / 构造和评测辅助，不是最终数据集外推理路线。
- VDT-CF-Attr no-true-context 是最终演示路线：推理只输入 `image + current_caption + VDT score`。
- 实验支持的结论是：可控反事实归因数据能训练一个不使用 true context 的错配类型 head；在加入额外原始 OOC different-event 样本后，真实 OOC 100 条评测有明显改善。
- 不要声称 MLP 在所有设置都最好；基础 1000 五类时 LR 更稳，plus2000 设置下 MLP 的类型准确率更高。
- 不要声称已经解决真实 OOC 泛化；真实 OOC 100 条结果显示方向正确但指标仍不高。
