# VDT-CF-Attr 结课报告 v8 修改说明

本文件记录第 8 版结课报告的最终修改口径。正式提交文件为：

- `10-VDT-CF-Attr-内容安全结课报告-v8.docx`
- `10-VDT-CF-Attr-内容安全结课报告-v8.pdf`

## 已完成的主要修改

1. 将报告主线统一为：VDT 主分类 + COVE-lite oracle 辅助 + VDT-CF-Attr no-true-context 最终应用路线。
2. 将 COVE-lite 从主方法降为 oracle / 构造与评测辅助，避免与数据集外推理目标冲突。
3. 重写摘要、关键词、引言、方法、实验分析、错误分析和结论。
4. 补写“已有工作与本组工作的边界”，明确 VDT、CLIP、BLIP-2、NewsCLIPpings 和 VisualNews 均为已有基础。
5. 增加 controlled counterfactual data、different-event 分布修正、group split 防泄漏、field-presence 后处理等方法细节。
6. 将实验结果分为 VDT baseline、no-true-context scaling、plus2000 synthetic held-out、真实 OOC 100 条评估四组，不混写 oracle 和 no-true-context。
7. 将真实 OOC 100 条结果降调表述为“明显改善但仍有挑战”，不写成已经解决真实场景归因。
8. 按课程要求扩写每位同学的工作、遇到的问题、解决过程、最终效果与收获。
9. 去掉口语化和泛泛表达，避免“完全解决”“显著领先”等过度结论。
10. 静态目录已填好，避免提交后出现空目录页。

## v8 报告中的关键结论

- VDT strict baseline 在两组目标域完成核心复现。
- no-true-context attribution head 随反事实训练规模增加而提升。
- 加入 different-event 训练后，真实 OOC 100 条评估的 Type Acc 从 0.0900 提升到 0.2900，Field Micro-F1 从 0.3276 提升到 0.4781。
- Exact Match 仍为 0.0300，说明真实 OOC 多字段复合错配仍未被完全解决。
- 最终结论限定为：项目完成了可训练、可评测、可演示的 no-true-context 错配归因框架，真实场景泛化仍需更多视觉证据与人工标注支撑。
