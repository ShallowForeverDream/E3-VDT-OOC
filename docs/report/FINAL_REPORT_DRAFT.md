# 期末报告初稿：基于事件语境与证据约束的跨域图文内容挪用检测系统

> 用途：报告负责人可直接以此为正文初稿，再按课程模板调整格式、插入图表和参考文献。

## 摘要

跨域图文内容挪用（Out-of-Context, OOC）是指真实图像被放入错误新闻语境中传播，造成误导性叙事。相比传统篡改检测，OOC 内容通常不修改图像像素，而是通过文本、时间、地点或事件语境错配制造虚假信息，因此更难依靠低层视觉伪造痕迹发现。

本项目围绕新闻场景下的图文内容挪用检测，首先在 NewsCLIPpings / VisualNews 数据上复现 VDT baseline，并完成 strict BLIP-2/GaussianBlur 设置下的特征预处理和跨域训练。在 `target_domain=bbc,guardian` 设置下，当前复现得到 `F1=0.7353`、`Acc=0.7383`、`AUC=0.7398`。在此基础上，我们提出 E3-VDT（Event-grounded and Explanation-enhanced VDT）系统路线：在不降低 VDT 主分类准确率的前提下，以 sidecar 方式增加事件字段一致性建模，输出 `mismatch_type`、`conflict_fields`、`event_scores` 和结构化解释，使系统不仅判断“是否错配”，还能回答“哪里错、为什么错、属于哪类错配”。

实验与系统实现表明，VDT 可以作为有效二分类 baseline，而 E3-VDT 的主要贡献在于提升内容安全审核场景中的可解释性、可诊断性和演示可用性。项目最终提供可运行 Gradio 展示系统、统一 JSON 输出 schema、复现实验日志和答辩样例。

## 1. 引言

社交媒体和新闻传播中，大量虚假信息并非来自图像本身的像素篡改，而是来自图文语境错配。例如，一张旧抗议现场图片可能被描述为近期另一个国家的事件；一张灾害现场图片可能被用于错误城市或错误年份。这类内容具有两个特点：

1. 图像往往是真实的，传统篡改检测难以发现；
2. 文本和图像在主题上可能高度相似，简单图文相似度模型容易被误导。

因此，OOC 检测需要同时理解图像、文本和事件语境。本项目选择 VDT 作为主 baseline，复现其跨域 OOC 检测能力，并进一步面向内容安全审核需求加入事件字段归因。

## 2. 相关工作

### 2.1 NewsCLIPpings 与 OOC 数据集

NewsCLIPpings 基于 VisualNews 构造图文匹配与错配样本，是新闻 OOC 检测的重要数据集。其优点是规模较大、适合跨域实验；不足是主要提供二分类标签，缺少细粒度错配类型标注，例如地点错配、时间错配、主体错配等。

### 2.2 VDT baseline

VDT 关注跨域图文错配检测，利用视觉语言特征和 domain adaptation 思路提升跨域泛化能力。该方法适合作为本项目的核心二分类 baseline。本项目在本机复现 VDT strict BLIP-2/GaussianBlur 流程，并记录硬件约束下的 batch size 调整。

### 2.3 可解释 OOC 检测需求

对于内容安全系统，仅输出 OOC / Non-OOC 往往不足。审核人员更需要知道冲突来自人物、地点、时间、事件类型还是行为关系。因此，本项目将可解释输出作为系统创新点，而不是单纯追求模型分数微小提升。

## 3. 方法设计

## 3.1 总体框架

E3-VDT 采用 accuracy-preserving 设计：

```text
VDT baseline -> OOC / Non-OOC 主分类结果
Event sidecar -> mismatch_type / conflict_fields / event_scores / explanation
```

默认情况下：

```text
final_label = vdt_label
final_score = vdt_score
```

事件字段模块不覆盖主分类结果，因此分类 Accuracy / F1 与 VDT baseline 持平；系统额外获得结构化解释能力。只有当验证集证明融合策略不降低 Accuracy/F1 时，才允许事件分数参与最终分类。

## 3.2 事件字段一致性建模

我们将图文语境拆成五个事件字段：

| 字段 | 含义 | 示例 |
|---|---|---|
| entity | 人物、组织、主体 | Obama / Elon Musk |
| location | 地点 | Paris / London |
| time | 时间 | 2024 / Monday |
| event_type | 事件类型 | protest / disaster / sports |
| relation | 行为关系 | attack / meet / rescue |

形成事件一致性向量：

```text
C_event = [s_entity, s_location, s_time, s_event_type, s_relation]
```

当某个字段一致性显著低时，系统输出对应冲突字段。

## 3.3 弱监督错配类型构造

由于现有数据集通常缺少细粒度错配标签，我们设计弱监督构造规则：

| 标签 | 触发条件 |
|---|---|
| entity mismatch | 主体实体冲突 |
| location mismatch | 地点冲突 |
| temporal mismatch | 时间冲突 |
| event-type mismatch | 事件类型冲突 |
| relation mismatch | 动作关系冲突 |
| context omission | 证据不足或上下文缺失 |

该标签既可用于系统展示，也可作为后续训练 attribution head 的弱监督信号。

## 3.4 结构化输出 schema

系统输出统一 JSON：

```json
{
  "label": "OOC",
  "confidence": 0.85,
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "event_scores": {
    "entity": 0.5,
    "location": 0.0,
    "time": 1.0,
    "event_type": 1.0,
    "relation": 1.0
  },
  "explanation": "地点字段冲突：文本为 Paris，图像上下文为 London。"
}
```

相比 VDT baseline，该输出更适合内容审核和答辩展示。

## 4. 实验设计

## 4.1 数据与设置

- 数据集：NewsCLIPpings / VisualNews
- 特征设置：strict BLIP-2 / GaussianBlur
- 复现模型：VDT baseline
- 指标：Accuracy、F1、AUC、F1-real、F1-fake、EER
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU

## 4.2 VDT baseline 复现结果

| 设置 | 状态 | F1 | Acc | AUC | 说明 |
|---|---|---:|---:|---:|---|
| `target_domain=bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 | bs256 CUBLAS 失败，bs128 跑通 |
| `target_domain=usa_today,washington_post`, bs128 | failed_oom | - | - | - | Epoch 1 中途 CUDA OOM |
| `target_domain=usa_today,washington_post`, bs64 | running / partial | 0.8013 | 0.8017 | 0.8006 | 当前已跑出 5 个 validation blocks，最终结果待训练结束确认 |

说明：第二组 bs64 当前为运行中结果，最终报告需以训练结束后的 best-by-F1 解析结果为准。

## 4.3 E3-VDT 与 baseline 对比方式

由于本项目要求分类准确率不能降低，我们采用分层评价：

| 层级 | VDT baseline | E3-VDT sidecar | 目标 |
|---|---|---|---|
| 分类层 | OOC / Non-OOC | 沿用 VDT 结果 | Accuracy / F1 持平 |
| 归因层 | 无结构化输出 | mismatch type + conflict fields | 显著增强 |
| 解释层 | 弱解释 | event scores + explanation | 更适合审核 |

因此，E3-VDT 的主要提升不是牺牲分类性能换解释，而是在分类指标持平的前提下增加可解释输出。

## 4.4 Hard Negative 分析

我们设计以下高相似错配样例：

| 类型 | 示例 | 预期输出 |
|---|---|---|
| same-topic different-location | Paris protest vs London protest | location mismatch |
| same-location different-event | Paris hospital vs Paris football match | event-type mismatch |
| same-person different-time | 同一人物不同年份 | temporal mismatch |
| same-domain multi-field | fire in New York vs football in London 2019 | multi-field conflict |

这些样例用于证明系统能诊断错配原因，而不是只给一个二分类标签。

## 5. 系统实现

系统采用 Gradio 实现，仓库中提供：

- `demo/app.py`：网页展示系统
- `src/e3vdt/inference/pipeline.py`：统一推理管线
- `examples/demo_cases.jsonl`：演示样例
- `examples/reproduction_metrics.json`：复现实验指标
- `docs/OUTPUT_SCHEMA.md`：输出格式定义

网页端包含两个标签页：

1. OOC 检测演示：输入图片、文本和图像上下文，输出结构化判断；
2. 复现实验指标：展示 VDT baseline 复现状态和指标。

## 6. 结果讨论

当前 VDT baseline 已经在一组 domain 上完成严格流程复现，证明本项目不是仅做界面展示，而是完成了真实数据、特征预处理和模型训练。E3-VDT 的创新点主要体现在：

1. 不降低分类准确率：默认沿用 VDT 主分类；
2. 提供错配类型：从二分类扩展到细粒度归因；
3. 输出冲突字段：支持内容安全审核定位问题；
4. 支持 Hard Negative 诊断：更贴近真实传播场景。

## 7. 不足与未来工作

1. 当前展示系统中的事件抽取仍是轻量规则/heuristic，需要后续接入更强 NER、OCR、captioning 或多模态大模型；
2. 第二组 domain 复现仍在运行，最终报告需补充完整指标；
3. 弱监督错配标签需要更多人工抽样校验；
4. 若尝试 event score 与 VDT score 融合，必须通过验证集 gate，确保分类指标不下降。

## 8. 成员贡献写法

| 角色 | 贡献 |
|---|---|
| 组长 | 项目统筹、VDT 复现、系统集成、GitHub 维护、答辩主线 |
| 复现负责人 | 数据准备、BLIP-2/GaussianBlur 预处理、VDT baseline 训练、指标整理 |
| 系统负责人 | Gradio demo、推理接口、输出 schema、样例构造 |
| 报告负责人 | 报告撰写、PPT 制作、文献整理、图表排版 |

## 9. 结论

本项目围绕 OOC 图文内容挪用检测，完成了 VDT baseline 的核心复现，并提出 accuracy-preserving 的 E3-VDT 系统路线。在不降低 OOC / Non-OOC 分类准确率的前提下，系统进一步输出错配类型、冲突字段、事件一致性分数和结构化解释，从而提升内容安全审核场景中的可解释性和可用性。
