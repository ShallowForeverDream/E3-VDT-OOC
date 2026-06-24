# 2026-06-24 VDT strict BLIP-2/GaussianBlur reproduction: usa_today,washington_post

## 目标

在 strict BLIP-2/GaussianBlur 预处理数据上补跑第二组 VDT baseline，用于和 `bbc,guardian` 结果形成对照。

## Attempt A：batch_size=128

```powershell
cd D:\MY_PROJECT\OOC\ooc_repro_baselines\external\VDT
D:\0tools\conda-envs\vdt_py38\python.exe -m trainers.train_VDT --batch_size 128 --max_epochs 20 --target_domain usa_today,washington_post --base_model blip-2 --loss_type simclr
```

状态：failed_oom。

关键错误：

```text
RuntimeError: CUDA error: out of memory
```

失败位置：Epoch 1 约 `2770it`，尚未进入 validation，因此没有指标块。失败日志保留在：

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_usa_today_washington_post\train_stdout.log
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_usa_today_washington_post\train_stderr.log
```

## Attempt B：batch_size=64

```powershell
cd D:\MY_PROJECT\OOC\ooc_repro_baselines\external\VDT
D:\0tools\conda-envs\vdt_py38\python.exe -m trainers.train_VDT --batch_size 64 --max_epochs 20 --target_domain usa_today,washington_post --base_model blip-2 --loss_type simclr
```

- 状态：running
- 启动时间：2026-06-24 20:20 Asia/Shanghai
- PID：`66608`
- 处理：设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`，降低 batch size 到 64。

## 已确认加载信息

```text
src_excluded_topic: ['usa_today', 'washington_post', 'bbc']
tgt_excluded_topic: ['bbc', 'guardian']
source train dataset size: 567012, target train dataset size: 409920
source validation dataset size: 59163, target validation dataset size: 42931
```

## 当前日志路径

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs64_20ep_usa_today_washington_post\train_stdout.log
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs64_20ep_usa_today_washington_post\train_stderr.log
```

## 监控

```powershell
D:\MY_PROJECT\OOC\datasets\check_vdt_blip2_strict_training_usa_wp.ps1
```


## 当前阶段性指标（running）

截至当前训练仍在运行，已解析出 7 个 validation blocks。当前 best-by-F1：

| F1 | Acc | AUC | F1-real | F1-fake | confusion matrix |
|---:|---:|---:|---:|---:|---|
| 0.8013 | 0.8017 | 0.8006 | 0.7884 | 0.8134 | `[[15861,5031],[3481,18558]]` |

说明：该结果为 running partial，最终报告应在训练进程结束后再次运行 `scripts/parse_vdt_log.py` 并更新 final 指标。

## 待补充

- bs64 完成后的 best checkpoint epoch。
- F1 / Acc / AUC / EER / confusion matrix。
- 本地 checkpoint 归档路径。
