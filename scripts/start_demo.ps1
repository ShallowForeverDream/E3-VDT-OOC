param(
  [switch]$Install,
  [switch]$SkipChecks,
  [int]$Port = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "[E3-VDT-OOC] repo: $RepoRoot" -ForegroundColor Cyan
Write-Host "[E3-VDT-OOC] python: $(python --version)" -ForegroundColor Cyan

if ($Install) {
  Write-Host "[E3-VDT-OOC] installing dependencies..." -ForegroundColor Yellow
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-dev.txt
  python -m pip install -e .
}

if (-not $SkipChecks) {
  Write-Host "[E3-VDT-OOC] running pre-demo checks..." -ForegroundColor Yellow
  python scripts/check_project.py
  python scripts/run_demo_cases.py
  python scripts/check_accuracy_preserving.py
  python scripts/check_final_deliverables.py
}

if ($Port -gt 0) {
  $env:GRADIO_SERVER_PORT = [string]$Port
  Write-Host "[E3-VDT-OOC] launching Gradio with explicit port $Port." -ForegroundColor Green
} else {
  Remove-Item Env:GRADIO_SERVER_PORT -ErrorAction SilentlyContinue
  Write-Host "[E3-VDT-OOC] launching Gradio with auto port selection." -ForegroundColor Green
}
Write-Host "[E3-VDT-OOC] URL will be printed below, e.g. http://127.0.0.1:7860" -ForegroundColor Green
python demo/app.py
