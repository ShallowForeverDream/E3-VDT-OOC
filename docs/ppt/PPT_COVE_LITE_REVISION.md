# PPT 修改版提纲：COVE-lite 归因路线

> 用这个文件替换旧 PPT 中“E3-VDT Full 显著提升分类指标 / Evidence Gate / Event-Guided TTT 已完成”的表述。

## 第 1 页：标题

基于事件语境的跨域图文内容挪用检测与错配归因系统  
VDT Baseline + COVE-lite True-Context Attribution

## 第 2 页：问题背景

- OOC 风险来自图文语境错配，不一定来自图像篡改。
- 内容审核不仅要知道是否错配，还要知道错在哪里。

## 第 3 页：任务定义

输入：新闻图文对。  
输出：

```text
OOC / Non-OOC
mismatch_type
conflict_fields
event_scores
explanation
```

## 第 4 页：现有方法不足

- VDT：能做二分类，但不解释错配字段。
- 相似度方法：可能只利用表面相似度。
- OOC 数据集：通常没有细粒度错配类型标签。
- 生成式解释：如果没有证据支撑，难以验证。

## 第 5 页：文献启发

| 工作 | 启发 |
|---|---|
| VDT | 主分类 baseline |
| SNIFFER | 解释需要上下文或证据 |
| COVE | 先获得图像真实上下文，再判断 caption |
| MUSE | 需要防止 similarity shortcut |
| 归因类工作 | 解释需要人工评测 |

## 第 6 页：VDT 复现结果

| 设置 | F1 | Acc | AUC | 状态 |
|---|---:|---:|---:|---|
| bbc,guardian bs128 | 0.7353 | 0.7383 | 0.7398 | completed |
| usa_today,washington_post bs64 | 0.8032 | 0.8032 | 0.8028 | completed |

口径：完成 VDT strict 核心设置两组复现，不夸大为完整复现全部实验。

## 第 7 页：方法总览

```text
VDT baseline -> OOC / Non-OOC
COVE-lite true context -> 图像真实语境
Event attribution sidecar -> mismatch_type / conflict_fields / explanation
```

## 第 8 页：COVE-lite true context

```text
NewsCLIPpings sample
-> image_id
-> VisualNews original caption/title/context
-> true_image_context
```

然后比较：

```text
current_caption vs true_image_context
```

## 第 9 页：事件字段归因

字段：

```text
entity
location
time
event_type
relation
```

输出：

```json
{
  "mismatch_type": "location mismatch",
  "conflict_fields": ["location"],
  "event_scores": {"location": 0.1}
}
```

## 第 10 页：为什么需要人工评测

规则输出不是天然正确。  
我们构造人工归因集，标注：

```text
gold_mismatch_type
gold_conflict_fields
rationale
```

## 第 11 页：归因 baseline 对比

| 方法 | 说明 |
|---|---|
| majority | 永远预测最常见类型 |
| sampled | 随机采样类型 |
| text-only | 不看 true context |
| COVE-lite event rule | current caption vs true context 字段比较 |

## 第 12 页：归因指标

```text
mismatch_type_accuracy
conflict_field_micro_f1
conflict_field_macro_f1
exact_match_rate
```

只有 COVE-lite 归因超过简单 baseline，才写成有效贡献。

## 第 13 页：Demo 展示

展示：

- location mismatch；
- temporal mismatch；
- event-type mismatch；
- evidence insufficient；
- accuracy-preserving 模式。

## 第 14 页：当前完成情况

已完成：

- VDT 两组复现；
- Gradio demo；
- 结构化输出；
- COVE-lite 数据构造脚本；
- 归因评测脚本。

待本地运行：

- 生成 true-context pairs；
- 人工标注 50-100 条；
- 输出 attribution metrics。

## 第 15 页：贡献总结

实事求是版：

1. 完成 VDT strict 核心 baseline 复现；
2. 发现 VDT 不输出错配字段；
3. 实现 COVE-lite true-context attribution；
4. 设计人工归因评测协议；
5. 实现可展示系统。

## 第 16 页：结尾话术

> VDT 判断是否 OOC；我们的 COVE-lite attribution 进一步解释哪里错，并通过人工归因评测检验解释是否可靠。
