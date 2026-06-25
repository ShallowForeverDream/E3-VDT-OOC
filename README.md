# E3-VDT-OOC：跨域图文内容挪用检测与错配归因系统

本仓库用于《内容安全》课程项目：**基于事件语境的跨域图文内容挪用检测与错配归因研究**。

当前准确定位：

> VDT 负责 `OOC / Non-OOC` 主分类；本项目在 VDT 之上实现 **COVE-lite true-context attribution**，输出错配类型、冲突字段、事件分数和结构化解释。

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
3. **人工归因评测协议**：通过人工标注集计算 `mismatch_type_accuracy`、`conflict_field_micro_f1`、`macro_f1` 和 `exact_match_rate`。
4. **Accuracy-preserving sidecar**：默认继承 VDT 主分类结果，不让解释模块覆盖 `final_label`。

详见：

- [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)
- [`docs/LITERATURE_REVIEW_OOC_EXPLANATION.md`](docs/LITERATURE_REVIEW_OOC_EXPLANATION.md)
- [`docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md`](docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md)
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

## 队友先看什么

1. [`docs/PROJECT_NEXT_STEPS_COVE_LITE.md`](docs/PROJECT_NEXT_STEPS_COVE_LITE.md)
2. [`docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md`](docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md)
3. [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)
4. [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)
5. [`docs/SYSTEM_DEMO_ACCEPTANCE.md`](docs/SYSTEM_DEMO_ACCEPTANCE.md)
6. [`docs/ppt/PPT_COVE_LITE_REVISION.md`](docs/ppt/PPT_COVE_LITE_REVISION.md)
7. [`docs/DEFENSE_QA_COVE_LITE_ADDENDUM.md`](docs/DEFENSE_QA_COVE_LITE_ADDENDUM.md)
8. [`docs/VDT-COVE-Attr-系统全流程深度讲稿.md`](docs/VDT-COVE-Attr-系统全流程深度讲稿.md)
9. [`docs/VDT-COVE-Attr-模块方法深挖版.md`](docs/VDT-COVE-Attr-模块方法深挖版.md)

## 大文件约定

不要提交 VisualNews 原图、`origin.tar`、BLIP-2 权重、VDT checkpoint、`.pt/.npy/.pkl/.ckpt`。本地路径复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml` 后自行修改。

## 最稳答辩口径

> 我们完成了 VDT strict baseline 的核心复现。VDT 负责判断是否 OOC。我们的新增工作是 COVE-lite true-context attribution：利用图像原始上下文与当前 caption 的事件字段差异，输出错配类型和冲突字段，并通过人工归因评测检验解释是否可靠。
