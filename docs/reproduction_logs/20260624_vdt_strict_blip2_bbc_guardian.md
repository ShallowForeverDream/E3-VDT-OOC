# 2026-06-24 VDT strict BLIP-2/GaussianBlur reproduction

## 目标

严格复现 VDT 在 NewsCLIPpings 上的 BLIP-2/GaussianBlur 设置，target domain 为 `bbc,guardian`。

## 数据与预处理状态

- VisualNews 使用 `origin.tar` tar-offset 流式读取，未解压原图。
- BLIP-2 feature extractor：LAVIS `blip2_feature_extractor/pretrain`。
- 预处理 pair 数：`902674`。
- 预处理状态：完成。
- assemble 输出：`processed_data_blip2_strict`。
- VDT 当前 `NewsCLIPpings/processed_data` 已切换为 strict BLIP-2 junction。

## assemble 汇总

| split | train rows | test rows |
|---|---:|---:|
| person_sbert_text_text | 17,768 | 1,816 |
| semantics_clip_text_text | 516,072 | 54,164 |
| merged_balanced | 71,072 | 7,264 |
| semantics_clip_text_image | 453,128 | 47,288 |
| scene_resnet_place | 124,860 | 13,636 |

总 annotation rows：`1,307,068`。

## 官方 batch_size=256 尝试

命令：

```powershell
python -m trainers.train_VDT --batch_size 256 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

结果：第 1 个 epoch 前几个 batch 触发 CUDA/CUBLAS：

```text
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
```

处理：记录为复现偏差，改用 `batch_size=128` 继续。

## 当前运行：batch_size=128

命令：

```powershell
python -m trainers.train_VDT --batch_size 128 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

当前状态：运行中。

已确认数据集加载成功：

```text
source train dataset size: 409920
source validation dataset size: 42931
target train dataset size: 772980
target validation dataset size: 81237
```

## 本地日志路径

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian\train_stdout.log
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian\train_stderr.log
```

本机监控脚本：

```powershell
D:\MY_PROJECT\OOC\datasets\check_vdt_blip2_strict_training.ps1
```

## 待补充

- 20 epoch 最终指标。
- TTT 前后指标。
- 与 VDT 论文官方结果的偏差分析。
