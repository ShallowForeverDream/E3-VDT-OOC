# 答辩问答与统一口径：VDT-COVE-Attr 版

本页用于统一组内说法，避免把复现状态、创新点和系统边界讲乱。

## 1. 30 秒项目简介

我们做的是跨域图文内容挪用检测与错配归因系统。主分类采用 VDT baseline，负责判断图文对是否 OOC；我们的新增工作是 VDT-COVE-Attr：借鉴 COVE 的 context-first 思路，用 VisualNews 原始上下文作为图像真实语境，与当前 caption 做事件字段比较，输出错配类型、冲突字段、事件分数和结构化解释。

一句话：

> VDT 判断是否 OOC；VDT-COVE-Attr 解释哪里错，并通过人工归因评测验证解释是否可靠。

## 2. 老师可能问：你们严格复现论文了吗？

推荐回答：

> 我们完成了 VDT strict BLIP-2/GaussianBlur 核心设置下两组跨域 baseline 复现。由于本机 RTX 4060 Laptop GPU 显存有限，官方 batch size 256 和部分 batch size 128 设置不稳定，因此我们记录了 CUBLAS/OOM 问题，并将 batch size 调整到 128/64。我们不会夸大为“完整复现论文全部实验设置”，而是准确表述为“完成核心 strict setting baseline，并记录硬件约束下的复现偏差”。

可汇报数据：

| 设置 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

不要说：

- “我们完整复现了论文所有指标。”
- “我们已经超过论文结果。”
- “我们分类准确率显著超过 VDT。”

## 3. 老师可能问：你们是不是只是用了 VDT，然后加解释？

推荐回答：

> 是。我们不回避这一点。VDT 是主分类 baseline，我们没有声称替代 VDT 或提出新的主分类 SOTA。我们的贡献是在 VDT 二分类输出之外，增加一个 context-grounded attribution 模块。现有 VDT 只告诉审核人员 OOC / Non-OOC，但不说明错在哪里；我们用 VisualNews 原始上下文作为图像真实语境，将 current caption 与 true image context 拆成 entity、location、time、event_type、relation 五类字段比较，输出 mismatch_type 与 conflict_fields。

## 4. 老师可能问：你们解释怎么知道是对的？

推荐回答：

> 不能只靠规则说它对。因此我们设计人工归因评测集，人工标注 `gold_mismatch_type` 和 `gold_conflict_fields`，再比较 majority、sampled、text-only 与 COVE-lite event rule。只有当 COVE-lite 的 field-F1 和 type accuracy 高于简单 baseline，我们才把它作为有效解释贡献；否则会如实写失败分析。

## 5. 老师可能问：为什么不用 SNIFFER / COVE 原版？

推荐回答：

> SNIFFER 需要多模态大模型指令微调和外部工具链，复现成本较高；COVE 原版包含更复杂的视觉实体、网页证据和语言模型推理。我们采用的是 COVE-lite：利用 NewsCLIPpings / VisualNews 已有的原始图像上下文，完成可落地的 context-first attribution，适合课程项目算力和时间约束。

## 6. 老师可能问：为什么分类准确率不会降低？

推荐回答：

> 正式策略采用 baseline-preserving。VDT baseline 输出 `baseline_label` 和 `baseline_score`，VDT-COVE-Attr 只输出错配类型和冲突字段。代码里 `classification_policy="baseline_preserving"` 时，最终 `label = baseline_label`，因此分类结果和 VDT 完全一致，Accuracy/F1 持平。只有后续验证集证明融合事件分数不降低指标时，才允许 guarded fusion。

## 7. 老师可能问：数据集没有错配类型标签怎么办？

推荐回答：

> 这是本项目遇到的核心问题。NewsCLIPpings 主要提供 OOC/Non-OOC 二分类标签，缺少 location mismatch、entity mismatch 等细粒度错配类型。因此我们不把自动规则输出直接当作真值，而是构造人工归因评测集，评估弱归因标签是否可靠。

## 8. 老师可能问：Evidence Gate 和 Event-Guided TTT 做了吗？

推荐回答：

> 这两个是扩展方向。当前已完成的是 VDT baseline 复现、COVE-lite true-context attribution、归因评测脚本和可解释 demo。Evidence Gate 与 Event-Guided TTT 不作为当前已完成主贡献。

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

归因实验本地运行：

```powershell
.\scripts\run_cove_lite_attribution_experiments.ps1 `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -MaxRecords 500 `
  -EvalSampleN 50
```

## 10. 老师可能问：目前还有什么不足？

推荐回答：

1. COVE-lite 依赖 VisualNews metadata，coverage 需要实际统计；
2. 事件字段抽取仍是轻量规则，可接入更强 NER/OCR/captioning；
3. 人工归因集规模有限；
4. 目前没有声称主分类超过 VDT；
5. Evidence Gate、Event-Guided TTT、attribution head 训练属于后续工作。

## 11. 分工模板

| 角色 | 主要工作 |
|---|---|
| 组长 | 路线设计、VDT 复现、GitHub 维护、系统集成、答辩主线 |
| 复现负责人 | 数据准备、VDT strict baseline、日志和指标整理 |
| 创新模块负责人 | COVE-lite 上下文、弱归因标签、人工评测集 |
| 系统/报告负责人 | Gradio demo、报告、PPT、案例分析 |

## 12. 最稳结尾

> 我们完成了 VDT strict baseline 核心复现，并在此基础上实现 VDT-COVE-Attr。VDT 负责判断是否 OOC；COVE-lite attribution 负责解释哪里错。我们不夸大为主分类 SOTA，而是通过人工归因评测验证解释模块能否提供有效辅助。
