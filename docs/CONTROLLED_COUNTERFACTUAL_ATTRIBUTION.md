# Controlled Counterfactual Attribution：可控反事实错配归因实验

本文档说明本项目新增的“可控错配原因标签”路线。它的目的不是替代 VDT 主分类，而是为“错在哪里”提供可训练、可评测的监督信号。

## 为什么要做这一步

原始 NewsCLIPpings/OOC 样本只告诉我们图文是否错配，通常不直接给出“错配类型”。如果直接从原始 OOC 样本里猜 `entity / location / time / event_type`，会有两个问题：

1. 一条 OOC 可能同时人物、地点、事件都不一致，标签不干净；
2. 没有 gold attribution，就无法证明解释模块比规则或 NLI baseline 更好。

所以我们增加一条更严谨的训练路线：

> 从 Non-OOC 正样本出发，只替换 caption 里的一个字段，构造“单字段最小错配”的反事实样本。由于编辑过程可控，错配类型和冲突字段就是天然 gold label。

## 当前实现范围

脚本：`D:\MY_PROJECT\OOC\E3-VDT-OOC\scripts\data\build_controlled_counterfactuals.py`

当前支持四类样本：

| 类型 | 构造方式 | gold_mismatch_type | gold_conflict_fields |
|---|---|---|---|
| `none` | 保留 Non-OOC 原样 | `benign illustrative image` | `[]` |
| `entity_swap` | 替换 caption 中一个 PERSON/ORG/NORP 或标题式实体短语 | `entity mismatch` | `["entity"]` |
| `location_swap` | 替换 caption 中一个 GPE/LOC/FAC | `location mismatch` | `["location"]` |
| `time_swap` | 替换 caption 中一个年份 | `temporal mismatch` | `["time"]` |

实现上做了这些约束：

- 只从 `label=0` 的 Non-OOC 样本构造，避免把原始 OOC 的未知错配混进训练集；
- 每条反事实样本只替换一个 span；
- replacement pool 来自同一批 Non-OOC 样本；
- location/entity 优先保持 spaCy 实体类型一致，无法加载 spaCy 时有保守 fallback；
- time 当前只替换年份，避免复杂日期改写带来的语法噪声；
- 输出 `validation` 字段，记录被替换 span、替换值和单 span 编辑检查。

## 一键运行

前置：先跑通 COVE-lite 500 条上下文样本，至少存在：

```text
D:\MY_PROJECT\OOC\E3-VDT-OOC\outputs\cove_lite_context_pairs.jsonl
```

然后运行：

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC

powershell -ExecutionPolicy Bypass -File .\scripts\run_controlled_counterfactual_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 80 `
  -NliModel facebook/bart-large-mnli `
  -NliDevice 0
```

如果电脑没有 GPU 或只想快速检查链路：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_controlled_counterfactual_experiment.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -Python python `
  -MaxPerType 40 `
  -NoTransformers
```

## 输出文件

```text
outputs/counterfactual/counterfactual_generation_stats.json
outputs/counterfactual/controlled_counterfactual_all.jsonl
outputs/counterfactual/controlled_counterfactual_train.jsonl
outputs/counterfactual/controlled_counterfactual_val.jsonl
outputs/counterfactual/controlled_counterfactual_test.jsonl
outputs/counterfactual/controlled_counterfactual_*_events.jsonl
outputs/counterfactual/controlled_counterfactual_*_features.jsonl
outputs/counterfactual/attribution_head_model.pkl
outputs/counterfactual/attribution_head_metrics.json
outputs/report_tables_v2.md
```

## 当前一次本地结果

本地 `MaxPerType=80` 运行结果：

| Edit type | Generated/kept |
|---|---:|
| none / positive | 80 |
| location_swap | 80 |
| entity_swap | 80 |
| time_swap | 27 |

切分后：

| Split | Rows |
|---|---:|
| train | 186 |
| val | 40 |
| test | 41 |

归因测试集结果：

| Method | N | Type Acc | Field Micro-F1 | Exact Match |
|---|---:|---:|---:|---:|
| majority | 41 | 0.2927 | 0.0000 | 0.2927 |
| field-wise NLI | 41 | 0.6098 | 0.5714 | 0.6098 |
| attribution head MLP | 41 | 0.9268 | 0.9474 | 0.9512 |

解释口径要谨慎：这证明 attribution head 在“可控单字段反事实测试集”上明显优于 majority 和直接 NLI baseline；它还不能单独证明真实 OOC 样本上的泛化效果。真实 OOC 泛化需要继续标注 `D:\MY_PROJECT\OOC\E3-VDT-OOC\examples\attribution_eval_candidates_annotate.xlsx` 并运行人工 gold set 评测。

## 答辩时怎么说

推荐表述：

> VDT 提供 OOC/Non-OOC 主分类。我们新增的工作不是强行让 VDT 输出解释，而是构造了一个 sidecar attribution 模块。由于原始 OOC 数据集没有细粒度错配原因，我们先用 Non-OOC 样本做单字段可控反事实编辑，得到干净的 attribution gold label，再训练一个轻量 attribution head 去融合字段级 NLI、证据相关性和图结构对齐特征。当前实验显示，在可控反事实测试集上 attribution head 明显优于直接规则/NLI 归因；真实 OOC 泛化评测需要人工标注集进一步验证。

不建议过度表述：

- 不要说“已经解决所有真实 OOC 的错配解释”；
- 不要说“超过 VDT 主分类准确率”；
- 不要把 controlled counterfactual 当成真实新闻错配标注。

