# 实验计划


## 总体实验原则：Accuracy-preserving

主结果采用不降低分类准确率的策略：

1. `VDT baseline`：提供 OOC / Non-OOC 分类指标。
2. `E3-VDT sidecar`：沿用 VDT 的分类结果，因此 Accuracy/F1 与 baseline 持平；额外评估 mismatch type、conflict fields、event scores 和案例解释。
3. `E3-VDT fusion`：只有当验证集 Accuracy/F1 不低于 VDT baseline 时才作为最终方法，否则作为消融结果。

验收线：`Acc(E3-VDT) >= Acc(VDT)`，至少不能低于 baseline。

## 阶段 1：主 baseline 复现

VDT original、VDT + TTT；target domain 从 `bbc,guardian` 起步；指标为 Accuracy / Macro-F1 / F1-real / F1-fake / AUC。

当前进度：

| 设置 | 状态 | F1 | Acc | AUC | 说明 |
|---|---|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur, `target_domain=bbc,guardian`, `batch_size=128` | completed | 0.7353 | 0.7383 | 0.7398 | 官方 `batch_size=256` 在本机 CUDA/CUBLAS 报错，降到 128 后跑通。 |
| VDT strict BLIP-2/GaussianBlur, `target_domain=usa_today,washington_post`, `batch_size=128` | failed_oom | - | - | - | Epoch 1 中途 CUDA OOM，保留日志作为复现偏差。 |
| VDT strict BLIP-2/GaussianBlur, `target_domain=usa_today,washington_post`, `batch_size=64` | running_partial | 0.7995 | 0.8002 | 0.7988 | 当前 best-by-F1，训练仍在运行，结束后确认 final。 |

## 阶段 2：轻量 baseline

Text-only、Image-context-only、CLIP similarity、MUSE-lite similarity classifier、Simple fusion classifier。

## 阶段 3：E3-VDT 增强

VDT + Event Vector、VDT + Attribution Head、E3-VDT Full。

## 阶段 4：Hard Negative

same-topic different-event、same-person different-time、same-location different-event、same-event-type different-location、context omission。

## 阶段 5：原因分析

按错配类型、CLIP 相似度分桶、domain pair、典型错误案例分析。
