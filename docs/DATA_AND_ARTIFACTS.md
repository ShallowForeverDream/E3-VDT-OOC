# 数据与大文件管理

GitHub 只放代码、文档、小样例和结果模板，不放大文件。

禁止提交：VisualNews 原图、origin.tar、BLIP-2 权重、VDT checkpoint、`.pt/.npy/.pkl/.ckpt`、训练缓存、TensorBoard。

每个人复制：

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

然后按自己电脑路径修改。共享模型建议使用百度网盘/阿里云盘/HuggingFace Dataset/GitHub Release，并在 README 写下载地址和 SHA256。
