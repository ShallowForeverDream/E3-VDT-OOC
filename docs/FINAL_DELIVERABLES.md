# 期末项目最终交付清单

本页给组内成员和答辩前自检使用：确认仓库、系统、复现、报告、PPT 的交付边界一致。

## 1. 代码与展示系统

仓库地址：

```text
https://github.com/ShallowForeverDream/E3-VDT-OOC
```

队友拉取后运行：

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

生成课程提交包：

```bash
python scripts/make_submission_package.py
```

输出位于 `outputs/submission/`，会自动排除数据集、模型权重、checkpoint、缓存和本地临时文件。

网页端重点展示三个页：

1. `OOC 检测演示`：展示错配类型、冲突字段、event scores 和 JSON 输出。
2. `分类不降验证`：证明 sidecar 解释模块不覆盖 VDT baseline 主分类。
3. `复现实验指标`：展示 VDT strict BLIP-2/GaussianBlur baseline 指标。

## 2. 报告与 PPT

| 交付物 | 路径 | 用途 |
|---|---|---|
| 报告 Markdown 初稿 | `docs/report/FINAL_REPORT_DRAFT.md` | 便于 Git diff 和队友修改 |
| 报告 DOCX 初稿 | `docs/report/E3-VDT-OOC-结课报告初稿.docx` | 提交/排版基础 |
| PPT 内容稿 | `docs/ppt/PPT_CONTENT_DRAFT.md` | 逐页讲稿和修改依据 |
| 答辩 PPT 初稿 | `docs/ppt/E3-VDT-OOC-答辩PPT初稿.pptx` | 课堂展示基础 |

注意：DOCX 已通过结构检查；由于本机缺少 LibreOffice，未做 DOCX 渲染截图 QA。PPTX 已生成预览并检查主要页面排版。

## 3. 复现实验状态

当前可汇报的严格 baseline：

| 实验 | 状态 | F1 | Acc | AUC |
|---|---|---:|---:|---:|
| `bbc,guardian`, bs128 | completed | 0.7353 | 0.7383 | 0.7398 |
| `usa_today,washington_post`, bs128 | failed_oom | - | - | - |
| `usa_today,washington_post`, bs64 | running_partial | 0.8013 | 0.8017 | 0.8006 |

不要夸大为“完整复现论文全部设置”。准确表述是：

> 已完成 VDT strict BLIP-2/GaussianBlur 核心设置的一组完整 baseline，并正在补充第二组跨域设置；项目创新部分采用 accuracy-preserving sidecar，在不降低分类 Accuracy/F1 的前提下增强错配归因。

## 4. 创新点与验收口径

核心创新不是“模型分数小幅提升”，而是：

1. 事件字段一致性建模：entity / location / time / event_type / relation。
2. 弱监督错配类型构造：解决 OOC 数据集缺少细粒度 mismatch label 的问题。
3. 结构化错配归因输出：`mismatch_type`、`conflict_fields`、`event_scores`、`explanation`。
4. Accuracy-preserving sidecar：主分类继承 VDT baseline，解释模块不降低分类准确率。

代码级验收：

```bash
python scripts/check_accuracy_preserving.py
```

通过条件：

```text
baseline-preserving mode keeps final labels identical to VDT baseline.
```

总交付物自检：

```bash
python scripts/check_final_deliverables.py
```

通过条件：

```text
final deliverable check passed.
```

## 5. 答辩演示推荐顺序

1. PPT 第 1-4 页：问题背景、任务定义、baseline 局限。
2. PPT 第 5-8 页：VDT baseline、E3-VDT 框架、accuracy-preserving 策略、事件字段。
3. 打开网页 `OOC 检测演示`，依次展示：
   - 正例：`ex01_non_ooc_same_event`
   - 地点错配：`ex02_location_mismatch`
   - 主体错配：`ex04_entity_mismatch`
   - Hard Negative：`ex07_multi_field_hard_negative`
4. 打开网页 `分类不降验证`，展示：事件字段发现冲突，但最终 label 仍等于 VDT baseline label。
5. 打开网页 `复现实验指标`，说明 baseline 复现状态。
6. 回到 PPT 第 13-15 页：对比、系统运行、总结。

## 6. 答辩前仍需人工确认

- 把组员真实姓名、学号、分工填到学校要求的封面/文档中。
- 第二组 VDT 训练结束后，重新解析日志并更新 `examples/reproduction_metrics.json`。
- 如果教师要求提交 PDF，需要将 DOCX/PPTX 另存为 PDF 后再检查版式。
- 如需在课堂机运行，提前安装依赖并测试端口是否可用。
