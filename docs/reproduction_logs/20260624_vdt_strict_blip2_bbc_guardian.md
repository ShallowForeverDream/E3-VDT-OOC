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

## batch_size=128 运行结果

命令：

```powershell
python -m trainers.train_VDT --batch_size 128 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

状态：已结束。训练没有出现 `Traceback` / `CUDA error`。源码中 `earlystop_epochs = 5`，且 `without_progress += 1` 每轮都会执行，因此该 run 实际完成到 `Epoch 5` 后停止。保存的最佳 checkpoint 为 `epoch=1`。

已确认数据集加载成功：

```text
source train dataset size: 409920
source validation dataset size: 42931
target train dataset size: 772980
target validation dataset size: 81237
```

模型文件保存在本地，不提交 GitHub：

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines\external\VDT\saved_model\VDTNews.pt
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian\VDTNews_blip2_strict_bbc_guardian_bs128_best_epoch1.pt
```

文件大小约 `124.04 MB`，保存时间 `2026-06-24 18:07:38`。

## batch_size=128 验证指标

| checkpoint/run block | Accuracy at EER | EER | AUC | F1 | Acc | F1 real | F1 fake | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| best / epoch 1 | 0.3318 | 0.6682 | 0.7291 | **0.7230** | **0.7273** | **0.6907** | 0.7562 | `[[24733,16459],[5692,34353]]` |
| later block | 0.2759 | 0.7241 | 0.7276 | 0.7193 | 0.7255 | 0.6804 | **0.7594** | `[[23738,17454],[4848,35197]]` |
| later block | 0.2551 | 0.7450 | 0.7241 | 0.7158 | 0.7220 | 0.6765 | 0.7563 | `[[23608,17584],[4999,35046]]` |
| final block | 0.2688 | 0.7312 | 0.7276 | 0.7209 | 0.7257 | 0.6866 | 0.7560 | `[[24417,16775],[5511,34534]]` |

当前可汇报的 strict BLIP-2/GaussianBlur VDT baseline：`F1=0.7230`、`Acc=0.7273`、`AUC=0.7291`，target domain 为 `bbc,guardian`。

## 本地日志路径

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian\train_stdout.log
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian\train_stderr.log
```

本机监控脚本：

```powershell
D:\MY_PROJECT\OOC\datasets\check_vdt_blip2_strict_training.ps1
```

## 待补充 / 下一步

- 对照论文表格确认该 `bbc,guardian` 设置应比较的官方指标。
- 运行另外一个 target domain，用于形成至少 2 组 baseline 结果。
- 在 E3-VDT 系统侧接入该 baseline 结果展示，但不把 124MB 模型提交 GitHub。
- 可选：修复 early-stop 逻辑后做 extended run，作为“实现修正/消融实验”，不要混作官方复现。

