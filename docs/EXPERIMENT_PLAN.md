# 实验计划

## 阶段 1：主 baseline 复现

VDT original、VDT + TTT；target domain 从 `bbc,guardian` 起步；指标为 Accuracy / Macro-F1 / F1-real / F1-fake / AUC。

## 阶段 2：轻量 baseline

Text-only、Image-context-only、CLIP similarity、MUSE-lite similarity classifier、Simple fusion classifier。

## 阶段 3：E3-VDT 增强

VDT + Event Vector、VDT + Attribution Head、E3-VDT Full。

## 阶段 4：Hard Negative

same-topic different-event、same-person different-time、same-location different-event、same-event-type different-location、context omission。

## 阶段 5：原因分析

按错配类型、CLIP 相似度分桶、domain pair、典型错误案例分析。
