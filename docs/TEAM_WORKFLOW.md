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
```

禁止提交数据集、模型权重、绝对路径配置、个人隐私和 token。
