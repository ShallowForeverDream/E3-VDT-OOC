# VDT 严格复现协议

## 当前目标

严格复现 VDT 在 NewsCLIPpings 上的 BLIP-2/GaussianBlur 官方指标。

## 当前本地关键路径

```text
VDT repo: D:/MY_PROJECT/OOC/ooc_repro_baselines/external/VDT
NewsCLIPpings repo: D:/MY_PROJECT/OOC/datasets/NewsCLIPpings_repo
VisualNews tar: E:/OOC_Datasets/VisualNews/origin.tar
Strict precompute script: D:/MY_PROJECT/OOC/ooc_repro_baselines/tools/make_vdt_blip2_strict_from_tar.py
Precompute state: D:/MY_PROJECT/OOC/ooc_repro_baselines/results/vdt_blip2_strict/precompute_state.json
```

## 预处理完成后

1. assemble 成 `processed_data_blip2_strict`；
2. 将 VDT 的 `NewsCLIPpings/processed_data` 指向 strict 输出；
3. 跑官方命令：

```powershell
cd D:/MY_PROJECT/OOC/ooc_repro_baselines/external/VDT
$env:PYTHONPATH='.'
python -m trainers.train_VDT --batch_size 256 --max_epochs 20 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
```

如果显存不足，记录偏差后改为 `--batch_size 128`。

## 每次实验记录

git commit、数据版本、batch size、seed、target domain、指标、是否 TTT、和论文指标差异、可能原因。
