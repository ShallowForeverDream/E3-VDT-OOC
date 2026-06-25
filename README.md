# E3-VDT-OOC：跨域图文内容挪用检测与错配归因系统

本仓库用于《内容安全》课程项目：**基于 VDT 的图文内容挪用检测与错配原因解释**。

## 当前最终定位

> **VDT 负责 `OOC / Non-OOC` 主分类；COVE-lite true-context attribution 作为 oracle / 构造与评测辅助；最终演示和应用路线是 `VDT-CF-Attr no-true-context head`，即推理阶段只输入 `image + current_caption`，不输入 `true_image_context`。**

本项目不声称已经超过 VDT 的主分类性能。当前贡献是：完成 VDT strict baseline 的核心复现，并把二分类结果扩展为可训练、可评测、可演示的错配归因系统。

---

## 1. VDT 主分类复现结果

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | failed: CUDA OOM | - | - | - |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

详见：[`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)。

---

## 2. 方法总览

### 2.1 主分类：VDT

```text
image + current_caption -> OOC / Non-OOC
```

VDT 只负责判断图文是否错配。解释模块默认继承 VDT 主分类结果，不覆盖 `final_label`。

### 2.2 训练数据：Controlled Counterfactual Attribution

原始 NewsCLIPpings 只有 OOC 二分类标签，没有错配原因标签。为训练归因模块，本项目从 Non-OOC 样本构造反事实样本：

```text
Non-OOC image-caption pair
  -> 只替换 caption 中一个实体 / 地点 / 年份字段
  -> 得到 entity / location / temporal mismatch gold label
```

进一步，根据 100 条真实 OOC 人工标注结果，真实错配以 `different-event mismatch` 为主，因此项目额外加入严格筛选的低相似原始 OOC 样本作为 `different-event mismatch` 训练样本，并排除人工 gold set，避免训练/评测泄漏。

### 2.3 最终推理：VDT-CF-Attr no-true-context head

最终推理阶段只允许：

```text
image + current_caption
```

系统会自动调用 `VDTAdapter` 得到 OOC / Non-OOC / Uncertain，再使用 no-true-context attribution head 输出：

```text
mismatch_type
conflict_fields
confidence
field_presence
postprocess_reason
```

`true_image_context` 只用于训练构造、oracle 上限和人工评估参考，不进入最终推理。

---

## 3. 当前核心创新点

1. **Controlled Counterfactual Attribution Data**  
   从 Non-OOC 样本构造单字段反事实错配，得到可控的错配原因标签。

2. **different-event training correction**  
   真实 OOC 100 条人工标注显示 `different-event mismatch` 占 85%，因此训练集中加入严格筛选的原始 OOC different-event 样本，修正训练分布。

3. **VDT-CF-Attr no-true-context head**  
   推理阶段不输入 true context，只使用 image、current caption、VDT score 和 CLIP prompt grounding 特征输出错配原因。

4. **Oracle / no-true-context 双评测**  
   COVE-lite true-context attribution 作为 oracle / 上限 / 构造辅助；no-true-context head 作为最终应用路线。

5. **真实 OOC 100 条人工评估集**  
   两批人工标注共 100 条，用于验证真实 OOC 场景下的归因泛化能力。

6. **Accuracy-preserving sidecar**  
   解释模块不覆盖 VDT 主分类标签，只增加错配原因、冲突字段和结构化解释。

---

## 4. 快速开始

```bash
git clone https://github.com/ShallowForeverDream/E3-VDT-OOC.git
cd E3-VDT-OOC
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_project.py
pytest -q
python scripts/check_final_deliverables.py
python demo/app.py
```

Windows 答辩现场一键启动：

```powershell
.\scripts\start_demo.ps1
```

队友只跑网页演示时不需要 VisualNews 原图、`origin.tar`、NewsCLIPpings 原始数据、VDT/BLIP-2 checkpoint。推荐使用导出的轻量 demo artifact：

```powershell
# 组长本机
powershell -ExecutionPolicy Bypass -File .\scripts\export_demo_artifact.ps1

# 队友本机
git pull origin main
powershell -ExecutionPolicy Bypass -File .\scripts\import_demo_artifact.ps1 `
  -ZipPath D:\path\to\e3-vdt-ooc-demo-artifact.zip
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1 -SkipChecks
```

详见：[`docs/TEAMMATE_REPRODUCTION.md`](docs/TEAMMATE_REPRODUCTION.md)。

---

## 5. 单页前端演示

当前 `demo/app.py` 已改为极简单页应用：

```text
输入区：图片上传 + 文字输入
输出区：一张结果卡片
```

结果卡片展示：

```text
OOC 判定
ooc 总可信度
错配类型
错配类型分数
冲突字段表
```

运行：

```powershell
python demo\app.py
```

单样本命令行推理：

```powershell
python scripts\infer\infer_vdt_cf_attr.py `
  --image examples\demo_images\london_climate_demonstration_monday.png `
  --caption "A large protest erupted in Paris on Monday after a new climate policy."
```

默认 `--vdt-label auto`，会自动调用 `VDTAdapter`，不需要手动输入 VDT label / score。

---

## 6. 关键实验结果

### 6.1 no-true-context scaling

本地使用 `outputs/cove_lite_context_pairs_3000.jsonl` 跑 `MaxPerType=80/200/1000`，三组泄漏检查均为 0。

| MaxPerType | Counts none/location/time/entity | Best stable method | Type Acc | Field Micro-F1 | Exact Match |
|---:|---|---|---:|---:|---:|
| 80 | 80/80/80/80 | logistic regression no-true-context | 0.2745 | 0.3564 | 0.1961 |
| 200 | 200/200/200/200 | logistic regression no-true-context | 0.4266 | 0.5195 | 0.2308 |
| 1000 | 1000/797/1000/1000 | logistic regression no-true-context | 0.5275 | 0.5719 | 0.3250 |

结论：no-true-context attribution head 随反事实训练数据增加而提升，但仍低于 true-context oracle。

### 6.2 plus2000 different-event setting

训练分布：

```text
none/entity/location/time/different-event = 1000/1000/1000/1000/3000
```

合成 held-out test：

| Method | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| majority | 0.4669 | 0.7012 | 0.4669 |
| logistic regression no-true-context | 0.3317 | 0.6655 | 0.4228 |
| image+caption MLP attribution head | **0.5220** | 0.6876 | 0.3487 |

### 6.3 真实 OOC 100 条人工评估

| Model | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|
| `no_true_context_attr_5way_1000` | 0.0900 | 0.3276 | 0.0300 |
| `no_true_context_attr_5way_plus2000` | **0.2900** | **0.4781** | 0.0300 |

结论：加入 additional original-OOC different-event 训练样本后，真实 OOC 100 条评估明显改善，但仍不能声称已经可靠解决真实 OOC 泛化。

---

## 7. 真实 OOC 人工标注集

导入两批中文人工标注表：

```powershell
python scripts\eval\import_real_ooc_manual_annotations.py
```

输出：

```text
examples/real_ooc_attribution_eval_set.jsonl
examples/real_ooc_manual_100_canonical.csv
outputs/real_ooc_manual_label_stats.json
examples/real_ooc_manual_100_summary.json
```

当前 100 条真实 OOC 人工标注显示：

```text
different-event mismatch 占 85%
冲突字段以 entity / event_type / relation / location / time 多字段复合冲突为主
```

这说明真实 OOC 并不是简单的单字段错配，因此本项目将人工分析结果反过来用于修正训练数据构造。

---

## 8. 重要文档

- [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)
- [`docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md`](docs/VDT_CF_ATTR_NO_TRUE_CONTEXT.md)
- [`docs/NO_TRUE_CONTEXT_SCALING_RESULTS.md`](docs/NO_TRUE_CONTEXT_SCALING_RESULTS.md)
- [`docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md`](docs/CONTROLLED_COUNTERFACTUAL_ATTRIBUTION.md)
- [`docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md`](docs/ATTRIBUTION_EXPERIMENT_PROTOCOL.md)
- [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)
- [`docs/TEAMMATE_REPRODUCTION.md`](docs/TEAMMATE_REPRODUCTION.md)

---

## 9. 大文件约定

不要提交 VisualNews 原图、`origin.tar`、BLIP-2 权重、VDT checkpoint、`.pt/.npy/.pkl/.ckpt`。`configs/paths.local.yaml` 只在重新跑完整实验时需要，网页演示不需要。

---

## 10. 最稳答辩口径

> 我们完成了 VDT strict baseline 的核心复现。VDT 负责 OOC / Non-OOC 主分类。解释部分分为两层：COVE-lite true-context attribution 作为 oracle / 构造与评测辅助；最终应用路线是 VDT-CF-Attr，用可控反事实样本训练不依赖 true context 的 image+caption attribution head。真实 OOC 100 条评估显示，加入 different-event 训练后指标明显改善，但真实 OOC 多字段复合错配仍然困难，因此我们不声称已经完全解决真实场景归因，而是证明了可训练、可评测、可演示的 no-true-context 归因框架。
