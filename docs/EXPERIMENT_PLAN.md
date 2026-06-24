# 实验计划

## 阶段 1：主 baseline 复现

VDT original、VDT + TTT；target domain 从 `bbc,guardian` 起步；指标为 Accuracy / Macro-F1 / F1-real / F1-fake / AUC。

当前进度：

| 设置 | 状态 | F1 | Acc | AUC | 说明 |
|---|---|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur, `target_domain=bbc,guardian`, `batch_size=128` | completed | 0.7353 | 0.7383 | 0.7398 | 官方 `batch_size=256` 在本机 CUDA/CUBLAS 报错，降到 128 后跑通。 |
| VDT strict BLIP-2/GaussianBlur, `target_domain=usa_today,washington_post`, `batch_size=128` | running | - | - | - | 第二组 domain 正在后台运行。 |

## 阶段 2：轻量 baseline

Text-only、Image-context-only、CLIP similarity、MUSE-lite similarity classifier、Simple fusion classifier。

## 阶段 3：E3-VDT 增强

VDT + Event Vector、VDT + Attribution Head、E3-VDT Full。

## 阶段 4：Hard Negative

same-topic different-event、same-person different-time、same-location different-event、same-event-type different-location、context omission。

## 阶段 5：原因分析

按错配类型、CLIP 相似度分桶、domain pair、典型错误案例分析。
