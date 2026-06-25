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

## 答辩口径

- VDT 主分类没有被修改；解释模块作为 sidecar，不覆盖 VDT 的 OOC / Non-OOC 输出。
- COVE-lite true-context 是 oracle / 上限 / 构造和评测辅助，不是最终数据集外推理路线。
- VDT-CF-Attr no-true-context 是最终演示路线：推理只输入 `image + current_caption + VDT score`。
- 实验支持的结论是：可控反事实归因数据能训练一个不使用 true context 的错配类型 head；在 1000 规模下 LR head 明显超过 prompt rule。
- 不要声称 MLP 已经最好；当前 MLP 没有稳定超过 LR。
- 不要声称已经解决真实 OOC 泛化；真实 OOC 仍需人工标注集验证。
