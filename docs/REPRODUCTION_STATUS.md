# Reproduction Status

本页记录课程项目中 VDT baseline 的严格复现进度。大数据集、缓存特征和模型权重只保存在本机，不提交 GitHub。

## 当前结论

| 实验 | 状态 | target_domain 参数 | F1 | Acc | AUC | 说明 |
|---|---|---|---:|---:|---:|---|
| VDT strict BLIP-2/GaussianBlur | completed | `bbc,guardian` | 0.7353 | 0.7383 | 0.7398 | `batch_size=128` 跑通；最佳 checkpoint 为 epoch 1。 |
| VDT strict BLIP-2/GaussianBlur | failed_oom | `usa_today,washington_post`, bs128 | - | - | - | Epoch 1 约 2770 iter 处 CUDA OOM，保留失败日志。 |
| VDT strict BLIP-2/GaussianBlur | completed | `usa_today,washington_post`, bs64 | 0.8032 | 0.8032 | 0.8028 | 已完成 13 个 validation blocks；best-by-F1 来自第 9 个 validation block。 |

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
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs128_20ep_bbc_guardian_bs128_ep20
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs64_20ep_usa_today_washington_post_bs64_ep20
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\VDTNews_bbc_guardian_bs128_ep20.pt
D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\VDTNews_usa_today_washington_post_bs64_ep20.pt
```

第二组已完成，不需要继续监控。解析日志：

```powershell
cd D:\MY_PROJECT\OOC\E3-VDT-OOC
python scripts\parse_vdt_log.py D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs64_20ep_usa_today_washington_post_bs64_ep20\train_stdout.log --out D:\MY_PROJECT\OOC\ooc_repro_baselines\results\vdt_blip2_strict\official_bs64_20ep_usa_today_washington_post_bs64_ep20\parsed_metrics.json
```

## 报告写法

- VDT 是 baseline：证明我们完成论文方法复现。
- E3-VDT 是系统创新：在二分类基础上进一步输出 `mismatch_type`、`conflict_fields`、`event_scores` 和解释。
- 两组 completed 结果均可放入最终实验表；bs128 OOM 作为硬件约束下的失败尝试保留。
