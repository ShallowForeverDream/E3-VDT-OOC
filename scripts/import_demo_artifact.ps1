param(
    [Parameter(Mandatory=$true)]
    [string]$ZipPath,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Join-Path $PSScriptRoot ".." }
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$ZipPath = (Resolve-Path $ZipPath).Path
Set-Location $ProjectRoot

Write-Host "Importing demo artifact:" -ForegroundColor Cyan
Write-Host "  $ZipPath"
Write-Host "Into repo:" -ForegroundColor Cyan
Write-Host "  $ProjectRoot"

Expand-Archive -Path $ZipPath -DestinationPath $ProjectRoot -Force

$head = Join-Path $ProjectRoot "outputs\no_true_context_attr_5way_plus2000\no_true_context_attr_head.pkl"
$cases = Join-Path $ProjectRoot "outputs\no_true_context_attr_demo_cases.jsonl"
$imgs = Join-Path $ProjectRoot "outputs\no_true_context_attr_demo_images"

if (-not (Test-Path $head)) { throw "Import finished but model head is missing: $head" }
if (-not (Test-Path $cases)) { throw "Import finished but demo cases are missing: $cases" }
if (-not (Test-Path $imgs)) { throw "Import finished but demo images are missing: $imgs" }

Write-Host "[OK] Demo artifact imported." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  python -m pip install -r requirements.txt"
Write-Host "  python -m pip install -e ."
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -SkipChecks"
