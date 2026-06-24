# 需要队友/组长提供的信息格式

## 实验结果

```text
实验名称：
代码分支：
运行命令：
数据版本：
模型配置：
随机种子：
指标：Accuracy / Macro-F1 / F1-real / F1-fake / AUC
日志路径：
结果文件路径：
遇到的问题：
```

## Demo 案例

```json
{
  "id": "case-001",
  "image_path": "本地路径或截图",
  "text": "新闻 caption",
  "image_context": "图像真实上下文/证据",
  "expected_label": "OOC",
  "expected_mismatch_type": "location mismatch",
  "why_this_case_is_good": "适合答辩展示的原因"
}
```

## 遇到问题时

提供运行命令、报错末尾 30 行、当前分支、Python 版本、期望结果、实际结果。
