# Run this after GitHub CLI is authenticated.
# If gh auth status says token is invalid, run:
#   gh auth logout -h github.com -u ShallowForeverDream
#   gh auth login -h github.com -w

$ErrorActionPreference = 'Stop'
$repo = 'ShallowForeverDream/E3-VDT-OOC'
$desc = 'E3-VDT: event-aware and explainable out-of-context image-text misinformation detection course project.'

gh auth status

# Create public repo if it does not exist.
$exists = $true
try { gh repo view $repo | Out-Null } catch { $exists = $false }
if (-not $exists) {
  gh repo create $repo --public --description $desc
}

# Ensure remote and push.
$remote = git remote get-url origin 2>$null
if (-not $remote) { git remote add origin "https://github.com/$repo.git" }
git push -u origin main

Write-Host "Pushed to https://github.com/$repo"
