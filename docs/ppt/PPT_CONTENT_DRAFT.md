# 答辩 PPT 内容稿（15 页）

> 用途：PPT 负责人可以直接按页复制到模板中；每页包含标题、核心内容和讲稿提示。

## 1. 标题页

**标题**：基于事件语境与证据约束的跨域图文内容挪用检测系统  
**副标题**：E3-VDT: Accuracy-preserving OOC Detection with Structured Mismatch Attribution

讲稿：我们关注新闻场景中的图文内容挪用，即真实图片被放入错误文本语境中传播的问题。

## 2. 问题背景

- 图像本身可能真实，没有像素篡改痕迹。
- 虚假性来自错误文本、地点、时间或事件语境。
- 内容安全审核不仅需要判断真假，还需要知道“错在哪里”。

讲稿：传统伪造检测更多关注图像是否被改过，但 OOC 的难点在于图像是真的，只是语境错了。

## 3. 任务定义

输入：新闻图片 + caption / claim  
输出：

```text
OOC / Non-OOC
mismatch_type
conflict_fields
event_scores
explanation
```

讲稿：我们的目标不是只输出一个二分类标签，而是输出结构化错配归因。

## 4. 现有方法不足

| 问题 | 表现 |
|---|---|
| 全局相似度 shortcut | 同主题不同事件容易误判 |
| 缺少细粒度标签 | 只知道错配，不知道哪里错 |
| 解释能力弱 | 不便于内容安全审核 |

讲稿：例如 Paris protest 和 London protest 语义很近，但地点不同，审核时必须指出这个冲突。

## 5. Baseline：VDT

- 使用 VDT 作为主二分类 baseline。
- 复现 NewsCLIPpings / VisualNews strict BLIP-2/GaussianBlur 设置。
- 提供 OOC / Non-OOC 分类能力。

讲稿：我们不是只做 demo，而是先复现论文 baseline，保证项目有实验基础。

## 6. 我们的方法：E3-VDT 总览

```text
VDT baseline -> 主分类结果
Event sidecar -> 错配类型 + 冲突字段 + 解释
```

关键原则：

> 分类准确率不下降，解释能力增强。

讲稿：E3-VDT 默认不覆盖 VDT 分类结果，而是在旁路增加解释模块。

## 7. Accuracy-preserving 策略

默认：

```text
final_label = vdt_label
final_score = vdt_score
```

只有验证集证明不降性能时，才启用融合：

```text
Acc(E3-VDT) >= Acc(VDT)
F1(E3-VDT) >= F1(VDT)
```

讲稿：这保证我们的创新不会牺牲分类准确率。

## 8. 创新点 1：事件字段一致性

五个事件字段：

| 字段 | 示例 |
|---|---|
| entity | Obama / Musk |
| location | Paris / London |
| time | 2024 / Monday |
| event_type | protest / sports |
| relation | attack / meet |

讲稿：我们把图文是否描述同一事件拆成多个可解释字段。

## 9. 创新点 2：弱监督错配类型

标签集合：

- entity mismatch
- location mismatch
- temporal mismatch
- event-type mismatch
- relation mismatch
- context omission

讲稿：现有数据集没有这些细粒度标签，因此我们基于事件字段冲突构造弱监督标签。

## 10. 创新点 3：结构化输出

示例输出：

```json
{
  "label": "OOC",
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "event_scores": {"location": 0.0, "time": 1.0}
}
```

讲稿：这比只输出 OOC 更适合内容安全审核和系统展示。

## 11. 实验设置

- 数据集：NewsCLIPpings / VisualNews
- 特征：BLIP-2 / GaussianBlur strict
- baseline：VDT
- 指标：F1、Accuracy、AUC、F1-real、F1-fake
- 硬件：RTX 4060 Laptop GPU

讲稿：由于本机显存限制，官方 batch size 256 未跑通，我们记录了偏差并降 batch size。

## 12. VDT 复现结果

| 设置 | F1 | Acc | AUC | 状态 |
|---|---:|---:|---:|---|
| bbc,guardian bs128 | 0.7353 | 0.7383 | 0.7398 | completed |
| usa_today,washington_post bs64 | 0.8013 | 0.8017 | 0.8006 | running / partial |

讲稿：第二组是运行中当前最优，最终结果需要训练结束后更新。

## 13. E3-VDT 与 VDT 对比

| 能力 | VDT | E3-VDT |
|---|---|---|
| 分类 | 有 | 有，默认持平 |
| 错配类型 | 无 | 有 |
| 冲突字段 | 无 | 有 |
| 事件分数 | 无 | 有 |
| 解释 | 弱 | 结构化 |

讲稿：我们的优化点是分类不降的前提下，增加可解释归因能力。

## 14. Demo 展示

推荐演示顺序：

1. Non-OOC 正常匹配；
2. location mismatch；
3. entity mismatch；
4. event-type mismatch；
5. multi-field hard negative；
6. evidence insufficient；
7. 切到“分类不降验证”，证明 sidecar 不覆盖 VDT baseline label。

讲稿：打开 JSON 输出，强调 mismatch_type、conflict_fields、event_scores，以及 decision_source=vdt_baseline。

## 15. 总结与分工

贡献总结：

1. 完成 VDT strict BLIP-2/GaussianBlur 核心复现；
2. 提出 accuracy-preserving 的 E3-VDT 系统路线；
3. 实现结构化错配归因输出；
4. 构建可演示 Gradio 系统和测试样例。

讲稿：一句话总结——VDT 判断是否错配，我们进一步解释哪里错，并且不降低分类准确率。
