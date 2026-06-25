# E3-VDT-OOC：跨域图文内容挪用检测与错配归因系统

本仓库用于《内容安全》课程项目：**基于事件语境的跨域图文内容挪用检测与错配归因研究**。

当前准确定位：

> VDT 负责 `OOC / Non-OOC` 主分类；本项目在 VDT 之上实现两层错配归因：**COVE-lite true-context oracle** 用于构造/评测，**VDT-CF-Attr no-true-context head** 用于更接近真实应用的 `image + caption` 推理。

本项目不声称当前版本已经超过 VDT 的主分类性能。当前主贡献是：完成 VDT strict baseline 核心复现，并将 VDT 的二分类输出扩展为可评测的错配归因系统。

## 当前复现结果

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | failed: CUDA OOM | - | - | - |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

详见：[`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)。

## 当前创新点

1. **VDT + COVE-lite true-context attribution**：利用 VisualNews 原始上下文作为图像真实语境，比较 current caption 与 true image context 的事件字段差异。
2. **事件字段归因**：输出 `entity / location / time / event_type / relation` 五类字段的一致性分数和冲突字段。
3. **Controlled Counterfactual Attribution**：从 Non-OOC 样本出发，只替换一个实体/地点/年份字段，构造带 gold mismatch type 的可控归因训练与测试集。
4. **VDT-CF-Attr no-true-context head**：训练阶段可用 true context 构造标签，但推理阶段 attribution head 只使用 `image + current_caption + VDT score`，不依赖 VisualNews 原始上下文。
5. **Oracle / no-true-context 双评测**：COVE-lite oracle 作为上限和评测辅助；no-true-context image+caption head 作为更贴近真实应用的最终路线。
6. **人工归因评测协议**：通过人工标注集计算 `mismatch_type_accuracy`、`conflict_field_micro_f1`、`macro_f1` 和 `exact_match_rate`。
7. **Accuracy-preserving sidecar**：默认继承 VDT 主分类结果，不让解释模块覆盖 `final_label`。

详见：

- [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)
- [`docs/LITERATURE_REVIEW_OOC_EXPLANATION.md`](docs/LITERATURE_REVIEW_OOC_EXPLANATION.md)
- [`docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md`](docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md)
- [`docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md`](docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md)
- [`docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md`](docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md)
- [`docs/PROJECT_NEXT_STEPS_COVE_LITE.md`](docs/PROJECT_NEXT_STEPS_COVE_LITE.md)

## 快速开始

```bash
git clone https://github.com/ShallowForeverDream/E3-VDT-OOC.git
cd E3-VDT-OOC
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_project.py
python scripts/run_cove_attr_demo_cases.py
python scripts/run_demo_cases.py
python scripts/check_accuracy_preserving.py
python scripts/check_final_deliverables.py
pytest -q
python demo/app.py
```

Windows 答辩现场一键启动：

```powershell
.\scripts\start_demo.ps1
```

## 运行 COVE-lite 归因实验

本实验需要本机已有 NewsCLIPpings JSON 和 VisualNews metadata。

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
git pull

.\scripts\run_cove_lite_attribution_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\metadata `
  -Python python `
  -EvalSampleN 120
```

如果 metadata 路径不确定：

```powershell
.\scripts\find_visualnews_metadata.ps1
```

生成文件：

```text
outputs/cove_lite_context_pairs.jsonl
outputs/weak_attribution_labels.jsonl
examples/attribution_eval_candidates.jsonl
```

人工标注候选集后保存为：

```text
examples/attribution_eval_set.jsonl
```

然后运行：

```powershell
python scripts/eval/run_attribution_baselines.py `
  --gold examples\attribution_eval_set.jsonl `
  --weak-labels outputs\weak_attribution_labels.jsonl `
  --output outputs\attribution_eval_metrics.json
```

## 运行可控反事实归因实验

用途：解决“原始 OOC 没有错配类型 gold label”的问题。脚本会从 Non-OOC 样本构造单字段最小错配，训练 attribution head，并把结果写入 `outputs/report_tables_v2.md`。

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC

powershell -ExecutionPolicy Bypass -File .\scripts\run_controlled_counterfactual_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 80 `
  -NliModel facebook/bart-large-mnli `
  -NliDevice 0
```

当前一次本地结果（`MaxPerType=80`，group split，test=38）：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.2368 | 0.2812 | 0.2368 |
| field-wise NLI | 0.7632 | 0.7200 | 0.6579 |
| attribution head MLP | 0.9474 | 0.9020 | 0.9211 |

泄漏检查结果：

```text
source_sample_id leakage = 0
image_id leakage = 0
text_id leakage = 0
cross-split duplicate caption = 0
```

注意：这是可控反事实测试集结果，不等同于真实 OOC 人工标注集泛化结果。真实 OOC 仍需要标注 `examples/real_ooc_attribution_eval_candidates.xlsx` 后评测。

常用后续命令：

```powershell
python scripts\eval\check_counterfactual_leakage.py --fail-on-leak

powershell -ExecutionPolicy Bypass -File .\scripts\run_counterfactual_scaling.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -Sizes 80,200,1000,3000

python scripts\eval\build_real_ooc_attribution_candidates.py `
  --output examples\real_ooc_attribution_eval_candidates.jsonl `
  --xlsx examples\real_ooc_attribution_eval_candidates.xlsx `
  --n 80
```

## 运行不依赖 true context 的 VDT-CF-Attr 实验

这是当前更接近真实应用的路线：训练时用反事实样本提供错配类型标签；推理/特征阶段只用图片、当前 caption 和 VDT 分数，不读 `true_image_context`。

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC

powershell -ExecutionPolicy Bypass -File .\scripts\run_no_true_context_attr_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 80 `
  -OutputDir outputs\no_true_context_attr `
  -Device cuda `
  -BatchSize 16
```

当前一次本地结果（`MaxPerType=80`，group split，test=38，CLIP image+caption features）：

| Method | Uses true context at inference? | Type Acc | Field Micro-F1 | Exact Match |
|---|---|---:|---:|---:|
| field prompt grounding rule | False | 0.3421 | 0.3415 | 0.3947 |
| logistic regression no-true-context | False | 0.4474 | 0.3939 | 0.2632 |
| image+caption MLP attribution head | False | 0.4474 | 0.3500 | 0.4474 |

解释：这个结果低于 COVE-lite oracle 是合理的，因为它不再使用真实上下文；它更能反映数据集外推理难度。

## 队友先看什么

1. [`docs/PROJECT_NEXT_STEPS_COVE_LITE.md`](docs/PROJECT_NEXT_STEPS_COVE_LITE.md)
2. [`docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md`](docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md)
3. [`docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md`](docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md)
4. [`docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md`](docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md)
5. [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)
6. [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)
7. [`docs/SYSTEM_DEMO_ACCEPTANCE.md`](docs/SYSTEM_DEMO_ACCEPTANCE.md)
8. [`docs/ppt/PPT_COVE_LITE_REVISION.md`](docs/ppt/PPT_COVE_LITE_REVISION.md)
9. [`docs/DEFENSE_QA_COVE_LITE_ADDENDUM.md`](docs/DEFENSE_QA_COVE_LITE_ADDENDUM.md)
10. [`docs/VDT-COVE-Attr-系统全流程深度讲稿.md`](docs/VDT-COVE-Attr-系统全流程深度讲稿.md)
11. [`docs/VDT-COVE-Attr-模块方法深挖版.md`](docs/VDT-COVE-Attr-模块方法深挖版.md)

## 大文件约定

不要提交 VisualNews 原图、`origin.tar`、BLIP-2 权重、VDT checkpoint、`.pt/.npy/.pkl/.ckpt`。本地路径复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml` 后自行修改。

## 最稳答辩口径

> 我们完成了 VDT strict baseline 的核心复现。VDT 负责判断是否 OOC。解释部分分两层：COVE-lite true-context attribution 作为 oracle/评测辅助；最终应用路线是 VDT-CF-Attr，用可控反事实样本训练不依赖 true context 的 image+caption attribution head。
