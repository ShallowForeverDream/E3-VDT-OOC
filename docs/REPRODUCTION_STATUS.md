# Reproduction Status

本页记录课程项目中 VDT baseline 的严格复现进度。大数据集、缓存特征和模型权重只保存在本机，不提交 GitHub。

## 当前结论

| 实验 | 状态 | target_domain 参数 | F1 | Acc | AUC | 说明 |
|---|---|---|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur | completed | `bbc,guardian` | 0.7353 | 0.7383 | 0.7398 | `batch_size=128` 跑通；最佳 checkpoint 为 epoch 1。 |
| VDT strict BLIP-2/GaussianBlur | failed_oom | `usa_today,washington_post`, bs128 | - | - | - | Epoch 1 约 2770 iter 处 CUDA OOM，保留失败日志。 |
| VDT strict BLIP-2/GaussianBlur | running | `usa_today,washington_post`, bs64 | - | - | - | 2026-06-24 20:20 已重启，等待完成后补指标。 |

## 已确认的数据规模

`bbc,guardian` run：

```text
source train dataset size: 409920
source validation dataset size: 42931
target train dataset size: 772980
target validation dataset size: 81237
```

`usa_today,washington_post` run：

```text
source train dataset size: 567012
target train dataset size: 409920
source validation dataset size: 59163
target validation dataset size: 42931
```

## 本地路径

```text
D:\MY_PROJECT\OOC\ooc_repro_baselines
esultsdt_blip2_strict\official_bs128_20ep_bbc_guardian
D:\MY_PROJECT\OOC\ooc_repro_baselines
esultsdt_blip2_strict\official_bs128_20ep_usa_today_washington_post
```

监控第二组：

```powershell
D:\MY_PROJECT\OOC\datasets\check_vdt_blip2_strict_training_usa_wp.ps1
```

解析日志：

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
python scripts\parse_vdt_log.py D:\MY_PROJECT\OOC\ooc_repro_baselines
esultsdt_blip2_strict\official_bs128_20ep_bbc_guardian	rain_stdout.log
```

## 报告写法

- VDT 是 baseline：证明我们完成论文方法复现。
- E3-VDT 是系统创新：在二分类基础上进一步输出 `mismatch_type`、`conflict_fields`、`event_scores` 和解释。
- 若第二组完成，就把两组结果一起放入实验表；若来不及完成，也可以作为“正在补充的复现实验”。
