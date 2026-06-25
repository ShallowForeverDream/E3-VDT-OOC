# 队友复现与降级链说明

## 1. 正确系统链路

最终应用页 `VDT-CF-Attr 无 true context` 的逻辑应是：

```text
image + current caption
  -> VDTAdapter 自动判断 OOC / Non-OOC / Uncertain
  -> 只有 OOC 才进入 VDT-CF-Attr 细粒度归因
  -> Non-OOC 直接输出 benign illustrative image
  -> Uncertain / 证据不足直接输出 uncertain，不强行归因
```

因此答辩时可以明确说：

> VDT 负责是否 OOC 的主判断；归因模块是 sidecar，只在 VDT 判断为 OOC 后解释“具体错在哪里”，不会覆盖 VDT 主分类。

## 2. 只从 GitHub 拉源码时会发生什么

仓库不提交以下本地产物：

```text
outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl
outputs/no_true_context_attr_5way_1000/image_caption_features_*.csv
outputs/no_true_context_attr_demo_images/
VisualNews 原图 / origin.tar
PyTorch / CLIP 权重缓存
```

所以队友只拉 GitHub 源码时，可能看到：

```text
vdt_adapter_insufficient_input
field_prompt_grounding_rule_fallback
evidence_status = uncertain
```

这不是“模型正常预测结果”，而是**安全降级链**：

1. 没有本地 `image_caption_features_*.csv`，VDTAdapter 不能加载轻量 feature-head；
2. 没装 `torch/transformers` 或没加载 CLIP，CLIP fallback 不能出图文相似度；
3. 没有 `no_true_context_attr_head.pkl`，归因头不能加载；
4. 系统返回 `Uncertain / evidence insufficient`，不强行编造错配类型。

注意：旧版本曾在 CLIP 全零时给出接近 1.0 的 benign 置信度。现在已修复：CLIP 不可用时会明确输出 `uncertain / insufficient visual evidence`。

## 3. 三档复现方式

### A. Source-only smoke test：只验证网页能跑

适合没有数据集、没有 GPU 的队友。

```powershell
git clone https://github.com/ShallowForeverDream/E3-VDT-OOC.git
cd E3-VDT-OOC
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\check_project.py
python demo\app.py
```

预期：

- `VDT-COVE-Attr 主系统`、`分类不降验证`、`复现实验指标`、`实验看板`能正常展示；
- `VDT-CF-Attr 无 true context` 如果没装 CLIP/没模型，会保守返回 `Uncertain`，这是正常降级，不是最终模型效果。

### B. Demo with CLIP fallback：能跑图文相似度，但没有训练头

适合想看自动 VDTAdapter 的队友。

建议使用 Python 3.10/3.11 的 conda 环境，然后安装 PyTorch + transformers：

```powershell
python -m pip install transformers
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果没有 CUDA，就安装 CPU 版 PyTorch。安装后运行：

```powershell
python demo\app.py
```

预期：

- `auto_vdt.decision_source` 可能为 `vdt_adapter_clip_similarity_fallback`；
- 如果仍没有 `no_true_context_attr_head.pkl`，归因部分还是 fallback，不代表训练头效果。

### C. Full local reproduction：完整训练头 + 真实演示效果

需要以下本地资源之一：

1. 队友自己拥有 NewsCLIPpings / VisualNews / origin.tar，并运行实验脚本生成 `outputs/`；
2. 组长把本地 demo artifact 包发给队友，队友解压到仓库根目录。

完整训练命令示例：

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
  -MaxDifferentEvent 1000
```

完整效果至少需要：

```text
outputs/no_true_context_attr_5way_1000/no_true_context_attr_head.pkl
outputs/no_true_context_attr_5way_1000/image_caption_features_train.csv
outputs/no_true_context_attr_5way_1000/image_caption_features_val.csv
outputs/no_true_context_attr_5way_1000/image_caption_features_test.csv
outputs/no_true_context_attr_demo_cases.jsonl
outputs/no_true_context_attr_demo_images/
```

这些文件不进 Git，是为了避免提交模型/数据大文件。

## 4. 常见输出怎么解释

| 输出 | 含义 | 是否正常 |
|---|---|---|
| `vdt_adapter_feature_head` | 使用本地特征表临时训练的自动 VDT head | 完整本机演示常见 |
| `vdt_adapter_clip_similarity_fallback` | 没有特征表，使用 CLIP 相似度 fallback | 可接受 demo fallback |
| `vdt_adapter_insufficient_input` | 没有可用图片特征或 CLIP 不可用 | source-only 正常 |
| `no_true_context_attr_head` | 成功加载训练好的归因头 | 完整模型效果 |
| `field_prompt_grounding_rule_fallback` | 归因头 `.pkl` 不存在，退到规则 | source-only 正常 |
| `vdt_non_ooc_gate` | VDT 判断 Non-OOC，不进入归因 | 正常 |
| `vdt_uncertain_gate` | VDT 不确定，不进入归因 | 正常且应这样 |

## 5. 答辩口径

如果队友电脑只有源码，不要把 fallback 输出当模型指标。正确说法是：

> GitHub 源码可以复现系统结构、UI、评估脚本和保守降级逻辑；完整模型效果需要本地训练产物或数据集生成的 `outputs/`。我们不把大模型权重、图片数据集和 `.pkl` 训练产物提交到 GitHub。

