# E3-VDT-OOC：跨域图文内容挪用检测系统

本仓库用于《内容安全》课程项目：**基于事件语境与证据约束的跨域图文内容挪用检测研究**。

项目目标不是只复现论文，而是实现一个可演示、可协作、可扩展的 OOC（Out-of-Context）图文内容挪用检测系统：

> 输入新闻图片与文本，输出 `OOC / Non-OOC` 判断、错配类型、冲突字段、事件一致性分数和结构化解释。

## 固定创新点

1. **事件字段一致性建模**：显式比较 entity/location/time/event-type/relation，避免只依赖全局图文相似度。
2. **弱监督错配类型构造**：针对现有 OOC 数据集缺少细粒度错配类型标签的问题，基于 NewsCLIPpings 构造方式、VisualNews 上下文、NER/规则生成弱标签。
3. **结构化错配归因输出**：输出 mismatch type、conflict fields、event scores 和 evidence-grounded explanation，便于答辩展示和量化评价。
4. **Hard Negative 评测协议**：专门评估 same-topic different-event、same-person different-time、same-location different-event 等高相似错配样本。

详见 [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)。

## 快速开始

```bash
git clone https://github.com/ShallowForeverDream/E3-VDT-OOC.git
cd E3-VDT-OOC
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_project.py
python scripts/run_demo_cases.py
python scripts/check_accuracy_preserving.py
python scripts/check_final_deliverables.py
pytest -q
python demo/app.py
```

生成课程提交包（自动排除数据集、模型权重、checkpoint、缓存等大文件）：

```bash
python scripts/make_submission_package.py
```

Windows 答辩现场一键启动：

```powershell
.\scripts\start_demo.ps1
```

网页无法打开时，可使用离线备用输出：

```bash
python scripts/export_demo_outputs.py
```

启动后终端会打印实际访问地址，例如 `http://127.0.0.1:7860`。如果 `7860` 被占用，程序会自动尝试 `7861`、`7862` 等后续端口；也可以手动指定：

```powershell
$env:GRADIO_SERVER_PORT="7861"
python demo/app.py
```

命令行推理：

```bash
python -m e3vdt.inference.cli --text "A protest erupted in Paris on Monday." --image-context "People gathered in London during an earlier climate demonstration in 2020."
```

正式实验的 accuracy-preserving / sidecar 调用方式：

```bash
python -m e3vdt.inference.cli \
  --policy baseline_preserving \
  --baseline-label OOC \
  --baseline-score 0.87 \
  --text "A protest erupted in Paris on Monday." \
  --image-context "People gathered in London during an earlier climate demonstration in 2020."
```

此时最终 `label` 会严格等于 VDT 的 `baseline_label`；事件模块只补充错配类型和解释，不会拉低分类准确率。

## 队友拉下来后先看什么

1. [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md)：项目一句话、背景、最终目标。
2. [`docs/INNOVATION_POINTS.md`](docs/INNOVATION_POINTS.md)：创新点，答辩时按这里讲。
3. [`docs/DIVISION_OF_WORK.md`](docs/DIVISION_OF_WORK.md)：每个人负责什么。
4. [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)：Git 分支、提交、PR 规则。
5. [`docs/WHAT_TO_SEND_ME.md`](docs/WHAT_TO_SEND_ME.md)：需要提供给组长/集成者的材料格式。
6. [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md)：系统输入输出格式，所有模块都按这个对齐。
7. [`docs/DEMO_CASES.md`](docs/DEMO_CASES.md)：系统测试与答辩演示样例。
8. [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md)：VDT baseline 复现进度和指标。
9. [`docs/ACCURACY_PRESERVING_STRATEGY.md`](docs/ACCURACY_PRESERVING_STRATEGY.md)：保证分类准确率不低于 baseline 的实现策略。
10. [`docs/report/FINAL_REPORT_DRAFT.md`](docs/report/FINAL_REPORT_DRAFT.md)：结课报告正文初稿。
11. [`docs/ppt/PPT_CONTENT_DRAFT.md`](docs/ppt/PPT_CONTENT_DRAFT.md)：答辩 PPT 逐页内容稿。
12. [`docs/FINAL_DELIVERABLES.md`](docs/FINAL_DELIVERABLES.md)：期末提交前总清单、演示顺序和仍需人工确认项。
13. [`docs/PRESENTATION_RUNBOOK.md`](docs/PRESENTATION_RUNBOOK.md)：答辩现场启动、备用方案和故障排查手册。
14. [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md)：答辩追问的统一回答口径。


## 分类准确率约束

本项目采用 **accuracy-preserving** 路线：主分类结果默认继承 VDT baseline，事件字段模块作为旁路输出错配类型、冲突字段和解释。只有当验证集证明融合策略不降低 Accuracy/F1 时，才启用事件分数参与最终分类。

## 分工

| 角色 | 主要任务 | 交付物 |
|---|---|---|
| 组长 | 统筹路线、GitHub 合并、最终答辩串联；同时参与复现和系统 | 周进度、最终 README、答辩主线 |
| 复现负责人 | VDT 严格复现、数据状态、实验日志、指标表 | `docs/reproduction_logs/`、结果 CSV |
| 系统负责人 | Gradio demo、推理接口、前后端接入 | `demo/`、`src/e3vdt/inference/` |
| 报告负责人 | 结课报告、PPT、图表、案例分析 | `docs/report/`、`docs/ppt/` |

## 大文件约定

不要提交 VisualNews 原图、`origin.tar`、BLIP-2 权重、VDT checkpoint、`.pt/.npy/.pkl/.ckpt`。本地路径复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml` 后自行修改。

## 当前复现状态

当前组长机器上已完成两组 VDT strict BLIP-2/GaussianBlur baseline；另保留一组 bs128 OOM 失败尝试作为硬件约束记录：

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian` | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | failed: CUDA OOM | - | - | - |
| `usa_today,washington_post`, bs64 | completed | 0.8032 | 0.8032 | 0.8028 |

详细路径、日志和复现协议见 [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md) 与 [`docs/REPRODUCTION_PROTOCOL.md`](docs/REPRODUCTION_PROTOCOL.md)。网页 demo 的“复现实验指标”标签页也会展示这些结果。


