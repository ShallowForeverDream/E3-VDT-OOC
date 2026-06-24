# GitHub 协作规范

## 分支

```text
main                稳定可展示版本
dev                 日常集成分支
feature/reproduce   VDT 复现和实验
feature/event       事件抽取与错配类型
feature/demo        展示系统
feature/report      报告和 PPT
```

## 基本流程

```bash
git checkout dev
git pull
git checkout -b feature/你的任务名
# 修改代码
git add .
git commit -m "feat: 简短说明"
git push origin feature/你的任务名
```

然后在 GitHub 上发 Pull Request 到 `dev`。

## Commit 前自检

```bash
python scripts/check_project.py
python scripts/run_demo_cases.py
python scripts/check_accuracy_preserving.py
python scripts/check_final_deliverables.py
pytest -q
```

## GitHub Actions CI

仓库已配置 `.github/workflows/ci.yml`，每次向 `main` / `dev` push 或发 PR 时会自动运行：

1. 安装 `requirements.txt` 和 `requirements-dev.txt`
2. 编译核心 Python 文件
3. 运行项目 smoke checks
4. 运行 demo 样例回归
5. 运行 accuracy-preserving 守护测试
6. 运行最终交付物自检
7. 运行 `pytest -q`

如果 CI 失败，先看失败的是哪一步：

- `run_demo_cases.py` 失败：说明 demo 输出和 `examples/demo_cases.jsonl` 期望不一致。
- `check_accuracy_preserving.py` 失败：说明 sidecar 可能覆盖了 VDT baseline 主分类，这是硬错误。
- `check_final_deliverables.py` 失败：说明报告/PPT/指标/CI/必要文件缺失或误提交了大文件。
- `pytest` 失败：说明接口 schema 或测试样例被破坏。

禁止提交数据集、模型权重、绝对路径配置、个人隐私和 token。

## 课程提交包

需要把项目打包发给老师或队友时运行：

```bash
python scripts/make_submission_package.py
```

输出在 `outputs/submission/` 下。该压缩包会排除数据集、模型权重、checkpoint、缓存和本地运行输出；GitHub 仓库仍然是主要协作入口。
