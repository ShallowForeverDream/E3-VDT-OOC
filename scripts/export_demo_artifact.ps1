param(
    [string]$ProjectRoot = "",
    [string]$OutputZip = "artifacts\e3-vdt-ooc-demo-artifact.zip",
    [switch]$IncludeLegacyFiveWay
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Join-Path $PSScriptRoot ".." }
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Set-Location $ProjectRoot

$zipPath = Join-Path $ProjectRoot $OutputZip
$stage = Join-Path $ProjectRoot "artifacts\demo_artifact_stage"

# Safety: stage must stay inside this repo's artifacts directory.
$artifactsRoot = Join-Path $ProjectRoot "artifacts"
New-Item -ItemType Directory -Force $artifactsRoot | Out-Null
if ((Resolve-Path $artifactsRoot).Path -notlike "$ProjectRoot*") {
    throw "Refusing to write outside project root: $artifactsRoot"
}
if (Test-Path $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force $stage | Out-Null

$required = @(
    "outputs\no_true_context_attr_5way_plus2000\no_true_context_attr_head.pkl",
    "outputs\no_true_context_attr_5way_plus2000\image_caption_features_train.csv",
    "outputs\no_true_context_attr_5way_plus2000\image_caption_features_val.csv",
    "outputs\no_true_context_attr_5way_plus2000\image_caption_features_test.csv",
    "outputs\no_true_context_attr_demo_cases.jsonl",
    "outputs\no_true_context_attr_demo_images"
)

$optional = @(
    "outputs\no_true_context_attr_5way_plus2000\no_true_context_attr_metrics.json",
    "outputs\no_true_context_attr_5way_plus2000\counterfactual_generation_stats.json",
    "outputs\no_true_context_attr_5way_plus2000\leakage_check.json",
    "outputs\real_ooc_no_true_context_eval_metrics.json",
    "outputs\report_tables_v2.md"
)

if ($IncludeLegacyFiveWay) {
    $optional += @(
        "outputs\no_true_context_attr_5way_1000\no_true_context_attr_head.pkl",
        "outputs\no_true_context_attr_5way_1000\image_caption_features_train.csv",
        "outputs\no_true_context_attr_5way_1000\image_caption_features_val.csv",
        "outputs\no_true_context_attr_5way_1000\image_caption_features_test.csv"
    )
}

function Copy-RelPath {
    param([string]$RelPath, [bool]$Required)
    $src = Join-Path $ProjectRoot $RelPath
    if (-not (Test-Path $src)) {
        if ($Required) { throw "Missing required demo artifact: $RelPath" }
        Write-Host "[skip optional] $RelPath" -ForegroundColor DarkYellow
        return
    }
    $dst = Join-Path $stage $RelPath
    $parent = Split-Path $dst -Parent
    New-Item -ItemType Directory -Force $parent | Out-Null
    $item = Get-Item $src
    if ($item.PSIsContainer) {
        Copy-Item -LiteralPath $src -Destination $parent -Recurse -Force
    } else {
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

foreach ($p in $required) { Copy-RelPath $p $true }
foreach ($p in $optional) { Copy-RelPath $p $false }

$manifest = @"
E3-VDT-OOC demo artifact
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

This package is for teammate demo only. It does NOT contain VisualNews origin.tar,
NewsCLIPpings raw data, BLIP-2/VDT checkpoints, or configs/paths.local.yaml.

How to use on teammate machine:
1. git pull origin main
2. powershell -ExecutionPolicy Bypass -File scripts\import_demo_artifact.ps1 -ZipPath <this zip>
3. python -m pip install -r requirements.txt
4. python -m pip install -e .
5. powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -SkipChecks

Expected behavior:
- VDT-CF-Attr page should load outputs/no_true_context_attr_5way_plus2000/no_true_context_attr_head.pkl first.
- No VisualNews/NewsCLIPpings path is required for demo inference.
- Full reproduction/training still requires the external datasets.
"@
$manifest | Set-Content -Path (Join-Path $stage "DEMO_ARTIFACT_README.txt") -Encoding UTF8

if (Test-Path $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Remove-Item -LiteralPath $stage -Recurse -Force

Write-Host "Demo artifact exported:" -ForegroundColor Green
Write-Host "  $zipPath"
Write-Host "Size: $sizeMb MB"
