# 答辩问答与统一口径

本页用于答辩前统一组内说法，避免把复现状态、创新点和系统边界讲乱。

## 1. 30 秒项目简介

我们做的是一个跨域图文内容挪用检测系统。传统 OOC 检测通常只输出“是否错配”，但内容安全审核更关心“哪里错、为什么错”。因此我们以 VDT 作为主分类 baseline，在不降低分类准确率的前提下，增加事件字段一致性分析，输出错配类型、冲突字段、事件分数和结构化解释。

一句话：

> VDT 判断是否错配，E3-VDT 进一步解释哪里错，并且不牺牲主分类准确率。

## 2. 老师可能问：你们严格复现论文了吗？

推荐回答：

> 我们完成了 VDT strict BLIP-2/GaussianBlur 核心设置下两组跨域 baseline 复现。由于本机 RTX 4060 Laptop GPU 显存有限，官方 batch size 256 跑不通，`usa_today,washington_post` 的 bs128 也出现 CUDA OOM，因此我们记录了 CUBLAS/OOM 问题并将 batch size 调整到 128/64。因此我们不会夸大为“完整复现论文全部实验设置”，而是准确表述为“完成核心 strict setting baseline，并记录硬件约束下的复现偏差”。

可汇报数据：

| 设置 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

不要说：

- “我们完整复现了论文所有指标。”
- “我们已经超过论文结果。”
- “把历史 running_partial 当成最终结果。”

## 3. 老师可能问：你们的创新点是什么？

推荐回答分三层：

1. **事件字段一致性建模**：把图文语义拆成 entity、location、time、event_type、relation，而不是只看全局相似度。
2. **错配类型归因输出**：系统输出 `mismatch_type`、`conflict_fields`、`event_scores` 和解释，能回答“哪里错、为什么错”。
3. **Accuracy-preserving sidecar**：主分类继承 VDT baseline，解释模块作为旁路输出，不覆盖主分类，从而保证分类 Accuracy/F1 不下降。

一句话：

> 我们的提升不是单纯刷分，而是把 OOC 检测从二分类扩展为可解释、可诊断、可展示的内容安全系统。

## 4. 老师可能问：为什么分类准确率不会降低？

推荐回答：

> 正式策略采用 baseline-preserving。VDT baseline 输出 `baseline_label` 和 `baseline_score`，E3 sidecar 只输出错配类型和冲突字段。代码里 `classification_policy="baseline_preserving"` 时，最终 `label = baseline_label`，因此分类结果和 VDT 完全一致，Accuracy/F1 持平。只有当验证集证明融合策略不降低指标时，才允许事件分数参与最终分类。

可以现场展示：

```bash
python scripts/check_accuracy_preserving.py
```

通过条件：

```text
baseline-preserving mode keeps final labels identical to VDT baseline.
```

## 5. 老师可能问：数据集没有错配类型标签怎么办？

推荐回答：

> 这是我们设计系统时遇到的核心问题。NewsCLIPpings 主要提供 OOC/Non-OOC 二分类标签，缺少 location mismatch、entity mismatch 等细粒度错配类型。因此我们采用弱监督构造思路：利用文本 caption、图像上下文、检索证据和规则/NER 抽取事件字段，再根据字段冲突生成弱 mismatch labels。当前 demo 用轻量规则展示这条路径，后续可以接入更强 NER、OCR、captioning 或人工标注校验。

强调：

- 细粒度错配类型是系统创新输出，不是原数据集直接给的字段。
- 当前 demo 是可解释系统原型，不把 heuristic 输出伪装成训练好的细粒度监督模型。

## 6. 老师可能问：你们的 demo 为什么要输入 image_context？

推荐回答：

> 当前网页 demo 为了避免“假装看懂图片”，要求输入图像上下文，例如原始 caption、OCR 或检索证据。严格复现部分使用 BLIP-2/GaussianBlur 特征作为 VDT baseline；展示系统则把 image_context 作为可解释归因模块的图像侧证据。后续接入 captioning/OCR 后，image_context 可以自动生成。

## 7. 老师可能问：VDT 和 E3-VDT 怎么比较？

推荐回答：

| 能力 | VDT | E3-VDT |
|---|---|---|
| OOC 二分类 | 有 | 有，默认继承 VDT |
| Accuracy / F1 | baseline | 持平，不降低 |
| 错配类型 | 无 | 有 |
| 冲突字段 | 无 | 有 |
| 事件分数 | 无 | 有 |
| 结构化解释 | 弱 | 有 |

核心说法：

> 分类层面我们尊重 VDT baseline；系统层面我们增强归因和可解释性。

## 8. 老师可能问：为什么不直接把事件分数融合进分类？

推荐回答：

> 直接融合可能提升某些样例解释效果，但也可能降低二分类 Accuracy/F1。因为本项目明确要求分类准确率不能降低，所以我们把 fusion 作为 guarded fusion：只有验证集满足 `Acc(E3-VDT) >= Acc(VDT)`、`F1(E3-VDT) >= F1(VDT) - tolerance` 时才启用；否则保持 sidecar。

## 9. 老师可能问：项目可运行性怎么证明？

推荐回答：

仓库提供完整运行和自检：

```bash
python scripts/check_project.py
python scripts/run_demo_cases.py
python scripts/check_accuracy_preserving.py
python scripts/check_final_deliverables.py
pytest -q
python demo/app.py
```

GitHub Actions 也会自动跑这些检查。答辩现场 Windows 可运行：

```powershell
.\scripts\start_demo.ps1
```

如果网页打不开，使用离线输出：

```bash
python scripts/export_demo_outputs.py
```

## 10. 老师可能问：为什么不把数据集和模型权重放 GitHub？

推荐回答：

> VisualNews 原图、BLIP-2 权重和 VDT checkpoint 都是大文件，不适合提交 GitHub。仓库提交的是代码、配置、文档、复现日志摘要和指标；大文件只保存在本机，并通过 `configs/paths.example.yaml` 说明本地路径配置方式。`check_final_deliverables.py` 会检查是否误提交模型或数据大文件。

## 11. 老师可能问：目前还有什么不足？

推荐回答：

1. 当前 demo 的事件抽取是轻量 heuristic，需要后续接入更强 NER/OCR/captioning。
2. 弱监督错配标签还需要人工抽样校验。
3. 若做真正的 attribution head，需要更多标注或更系统的弱标签评测。
4. 受限于本机 GPU，复现实验没有覆盖论文全部 domain/超参数组合。

## 12. 老师可能问：每个人怎么分工？

推荐回答模板：

| 角色 | 主要工作 |
|---|---|
| 组长 | 路线设计、GitHub 维护、系统集成、答辩主线 |
| 复现负责人 | 数据准备、VDT strict baseline、日志和指标整理 |
| 系统负责人 | Gradio demo、推理接口、JSON schema、样例构造 |
| 报告负责人 | 报告、PPT、文献和图表整理 |

如果不写真名，就按角色讲；提交版再按老师要求补真实姓名/学号。

## 13. 最稳妥的结尾话术

> 总结来说，我们完成了 VDT strict baseline 的核心复现，并在此基础上设计了 accuracy-preserving 的 E3-VDT 系统。它不以牺牲分类准确率换解释性，而是在保持 VDT 主分类结果的同时，输出错配类型、冲突字段、事件一致性分数和结构化解释，更适合内容安全审核和课堂展示。
