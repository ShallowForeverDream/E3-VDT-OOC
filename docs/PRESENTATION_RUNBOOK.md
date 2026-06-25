# 答辩现场运行手册

本手册用于课堂展示前 10 分钟自检，以及现场出现端口、依赖、网络问题时快速切换备用方案。

## 1. 推荐现场准备

答辩前在展示电脑上提前执行：

```powershell
git clone https://github.com/ShallowForeverDream/E3-VDT-OOC.git
cd E3-VDT-OOC
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_final_deliverables.py
```

如果已经拉过仓库：

```powershell
cd E3-VDT-OOC
git pull
python scripts/check_final_deliverables.py
```

## 2. 一键启动

Windows PowerShell：

```powershell
cd E3-VDT-OOC
.\scripts\start_demo.ps1
```

如果现场时间紧、不想跑完整检查：

```powershell
.\scripts\start_demo.ps1 -SkipChecks
```

如果依赖没有装：

```powershell
.\scripts\start_demo.ps1 -Install
```

启动后终端会打印访问地址，通常类似：

```text
http://127.0.0.1:7860
```

## 3. 推荐展示顺序

### 3.1 PPT 主线

1. 第 1-4 页：问题背景、任务定义、baseline 局限。
2. 第 5-8 页：VDT baseline 与 COVE-lite true-context attribution。
3. 第 9-10 页：Evidence relevance、field-wise NLI、人工 attribution 评测协议。
4. 切到网页 demo，优先展示 `VDT-COVE-Attr 主系统`。
5. 后续页面：复现指标、实验计划、边界与下一步。

### 3.2 网页 demo 主线

先看首页 dashboard：说明系统由 VDT 主分类、COVE-lite 真实语境、evidence relevance、field-wise NLI attribution 组成。

在 `VDT-COVE-Attr 主系统` 标签页依次展示：

1. `route01_location_time`：Paris claim vs London/2019 true context，展示 location + time conflict。
2. `route03_entity`：Obama claim vs Musk true context，展示 entity conflict。
3. `route06_evidence_insufficient`：true context 为空，展示 evidence sufficiency gate 不强行解释。
4. `route08_multi_field`：多字段 hard negative，展示 location/time/event_type 同时冲突。

然后切到 `分类不降验证`：

- `VDT baseline label` 故意选 `Non-OOC`。
- 点击“验证 sidecar 不覆盖主分类”。
- 讲解：归因模块可以发现字段冲突，但最终 `label` 仍继承 VDT baseline，因此不降低 VDT 主分类准确率。

最后切到 `复现实验指标` 和 `实验看板`：

- 说明 `bbc,guardian` 已完成 strict baseline：F1=0.7353，Acc=0.7383，AUC=0.7398。
- 说明 `usa_today,washington_post` bs64 已完成：F1=0.8032，Acc=0.8032，AUC=0.8028；bs128 OOM 作为硬件约束记录。
- 说明归因大规模实验脚本已准备，答辩后按人工 gold set + ablation 补最终实验表。

## 4. 备用方案

### 4.1 网页打不开

先看终端真实 URL。若端口冲突：

```powershell
.\scripts\start_demo.ps1 -Port 7861
```

或直接：

```powershell
$env:GRADIO_SERVER_PORT=7861
python demo/app.py
```

### 4.2 Gradio 依赖缺失

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python demo/app.py
```

### 4.3 现场不能联网

本项目 demo 不依赖联网。若不能重新安装依赖，使用已经生成的备用输出：

```powershell
python scripts/export_demo_outputs.py
type examples\demo_outputs.json
```

也可以打开：

```text
docs/DEMO_CASES.md
examples/demo_outputs.json
docs/ppt/E3-VDT-OOC-答辩PPT初稿.pptx
```

用这些文件展示样例输入、预期输出和实际 JSON。

### 4.4 Python 环境混乱

优先使用一个干净环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
python scripts/check_project.py
python demo/app.py
```

### 4.5 教师问“你们是不是牺牲准确率换解释”

回答：

> 不是。正式实验采用 accuracy-preserving sidecar。VDT baseline 负责主分类，E3 模块只输出错配类型和冲突字段。代码里 `classification_policy="baseline_preserving"` 时，最终 `label` 严格等于 `baseline_label`，所以分类 Accuracy/F1 与 VDT 持平。只有验证集证明不降指标时，才考虑融合。

可以现场展示：

```powershell
python scripts/check_accuracy_preserving.py
```

## 5. 答辩前最终检查

```powershell
python scripts/check_project.py
python scripts/run_cove_attr_demo_cases.py
python scripts/run_demo_cases.py
python scripts/check_accuracy_preserving.py
python scripts/check_final_deliverables.py
pytest -q
python scripts/make_submission_package.py
```

全部通过后，提交包在：

```text
outputs/submission/
```

## 6. 当前需要诚实说明的边界

- 不声称完整复现论文全部设置；当前已完成两组 core strict setting baseline，并记录 bs128/OOM 等硬件约束。
- 91GB 原图、BLIP-2 权重、VDT checkpoint 不提交 GitHub，只记录路径和日志。
- 当前可验收系统已完成 VDT-COVE-Attr 演示闭环；大规模归因有效性仍需人工 gold set 和 ablation 支撑，不能把 curated demo 指标当最终论文实验。
