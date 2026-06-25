# 当前任务看板

## P0：必须完成

- [x] VDT strict BLIP-2/GaussianBlur 预处理完成
- [x] assemble processed_data
- [x] 跑 VDT 官方复现（bbc/guardian 与 usa_today/washington_post bs64 已完成；bs256/bs128 显存问题已记录）
- [x] 建立 GitHub 项目结构
- [x] 建立 demo 初版
- [x] 固定输出 schema
- [x] 写入最终实验结果表

## P1：重要

- [~] mismatch type 弱标签脚本接入 NewsCLIPpings（demo/sidecar 已有规则版，待批量化）
- [x] hard negative 子集构造（demo cases 已覆盖核心类型，待扩展为评测集）
- [ ] CLIP/MUSE-lite baseline
- [x] 报告初稿
- [x] PPT 初稿

## P2：加分

- [ ] Event Vector 融入 VDT classifier
- [ ] Event-guided TTT
- [ ] Evidence Gate
- [ ] 自动生成案例图表
