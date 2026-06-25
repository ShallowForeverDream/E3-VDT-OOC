# 技术路线答辩口径

## 1 分钟版

我们的任务是新闻图文内容挪用检测。它不同于 deepfake：图片本身可能是真实的，文本本身也可能看似合理，但图片被放到了错误新闻语境里。我们首先复现 VDT 作为 OOC 二分类 baseline，VDT 能判断是否 OOC，但不能解释错在哪里。因此我们提出 VDT-COVE-Attr v2：保留 VDT 主分类结果，在后面增加一个 context-grounded attribution sidecar。具体做法是借鉴 COVE 的 context-first 思路，用 VisualNews 原始上下文构造图片真实语境，再对 current caption 和 true image context 分别抽取事件字段，通过 field-wise NLI 判断 entity、location、time、event_type、relation 哪些字段存在 contradiction。最后输出 mismatch_type 和 conflict_fields。归因效果不靠主观展示，而通过人工标注 attribution eval set，计算 type accuracy 和 field-F1 验证。

## 5 分钟版

1. **任务背景**：OOC misinformation 是真实图像被错误文本挪用，低成本、迷惑性强。
2. **Baseline**：我们复现 VDT，两组 strict BLIP-2/GaussianBlur baseline 已完成。VDT 解决是否 OOC，但没有错配归因。
3. **问题拆解**：可解释性拆为两个子问题：事件抽取、上下文归因。
4. **已有成果迁移**：COVE 告诉我们先找图片真实上下文；RED-DOT/CMIE 告诉我们要过滤无关证据；MUSE 提醒相似度 shortcut 很强；AMG 说明 attribution 需要人工标签评测。
5. **技术路线**：VDT 输出主标签；VisualNews metadata 提供 true context；事件抽取器生成 event tuples；field-wise NLI 判断字段矛盾；evidence relevance 控制是否强行解释。
6. **实验设计**：VDT baseline 复现、context coverage、事件抽取 F1、归因 baseline 对比、hard negative 评估。
7. **预期贡献**：不是替代 VDT，而是把 VDT 从黑盒二分类扩展为有证据约束、可评测的错配归因系统。

## 老师可能问：你们是不是只做规则？

答：当前仓库原型中的规则 sidecar 只作为 baseline。最终技术路线已升级为 field-wise NLI + evidence relevance。我们会把 rule extractor、similarity-only、majority/random 都作为对比方法，主张只在人工 gold attribution set 上由 field-F1 和 type accuracy 支撑。

## 老师可能问：你们有没有真实归因标签？

答：NewsCLIPpings 原始数据只有 OOC 二分类标签，没有 mismatch_type 或 conflict_fields。因此我们不把弱标签当真实标签。我们会构造小规模人工 attribution eval set，用它评估各归因方法。

## 老师可能问：是否提升 VDT 分类准确率？

答：我们的主目标不是替代 VDT 主分类，而是补充 VDT 缺失的解释能力。采用 sidecar 方式保留 VDT 输出，分类性能不被解释模块破坏。后续如果归因模块稳定，可以探索把 field-NLI features 接入 VDT 主干。

## 老师可能问：和 COVE/SNIFFER 有什么区别？

答：SNIFFER 做 MLLM instruction tuning，成本高；COVE 预测图像真实上下文。我们利用 NewsCLIPpings 与 VisualNews 的数据关系，直接获得图像原始上下文，构造轻量 COVE-lite，并接到已复现 VDT 后面，重点解决 VDT 缺少结构化归因的问题。
