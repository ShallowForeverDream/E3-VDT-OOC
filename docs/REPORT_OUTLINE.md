# 结课报告结构

对应课程要求：Title / Abstract / Introduction / Proposed Scheme / Experiment / Conclusion / 成员工作。

1. Title：基于事件语境与证据约束的跨域图文内容挪用检测系统
2. Abstract：问题、方法、实验、系统、贡献
3. Introduction：OOC 定义、内容安全意义、问题挑战
4. Related Work：NewsCLIPpings、VDT、SNIFFER、COVE、MUSE、Event-Radar
5. Proposed Scheme：VDT baseline、Event Consistency Vector、Weak Mismatch Labels、Attribution Head、Demo architecture
6. Experiment：数据集、复现设置、baseline、主结果、hard negative、消融、案例分析
   - 已完成 baseline：VDT strict BLIP-2/GaussianBlur，`target_domain=bbc,guardian`，`F1=0.7353`，`Acc=0.7383`，`AUC=0.7398`。
   - 正在补充 baseline：VDT strict BLIP-2/GaussianBlur，`target_domain=usa_today,washington_post`。
   - 对比重点：VDT 输出二分类；E3-VDT 输出二分类 + mismatch type + conflict fields + event scores + explanation。
7. System Implementation：Gradio 界面、输入输出 schema、后端接口、示例展示
8. Conclusion：贡献、不足、未来工作
9. Member Contribution：按角色写清楚每个人做了什么
