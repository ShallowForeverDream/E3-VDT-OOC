# GitHub 推送说明

本地仓库已经完成并提交。若 GitHub CLI 凭据有效，运行：

```powershell
scripts/push_to_github_after_login.ps1
```

如果出现 `Bad credentials`，先重新登录：

```powershell
gh auth logout -h github.com -u ShallowForeverDream
gh auth login -h github.com -w
```

浏览器完成授权后，再运行：

```powershell
scripts/push_to_github_after_login.ps1
```

目标仓库：

```text
https://github.com/ShallowForeverDream/E3-VDT-OOC
```
