param(
    [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
    [string]$Python = "python",
    [int[]]$Sizes = @(80, 200, 1000, 3000),
    [string]$NliModel = "facebook/bart-large-mnli",
    [int]$NliDevice = 0,
    [string]$OutputRoot = "outputs\counterfactual_scaling",
    [switch]$NoCopySummary,
    [switch]$NoTransformers
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

$rows = @()
foreach ($size in $Sizes) {
    $outDir = Join-Path $OutputRoot ("max_" + $size)
    Write-Host "===== scaling MaxPerType=$size output=$outDir ====="
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ".\scripts\run_controlled_counterfactual_experiment.ps1",
        "-ProjectRoot", $ProjectRoot,
        "-Python", $Python,
        "-MaxPerType", "$size",
        "-NliModel", $NliModel,
        "-NliDevice", "$NliDevice",
        "-OutputDir", $outDir,
        "-SkipReportTables"
    )
    if ($NoTransformers) { $args += "-NoTransformers" }
    powershell @args
    if ($LASTEXITCODE -ne 0) { throw "controlled counterfactual run failed for MaxPerType=$size" }

    & $Python "scripts\eval\check_counterfactual_leakage.py" `
        --train "$outDir\controlled_counterfactual_train.jsonl" `
        --val "$outDir\controlled_counterfactual_val.jsonl" `
        --test "$outDir\controlled_counterfactual_test.jsonl" `
        --output "$outDir\leakage_check.json"
    if ($LASTEXITCODE -ne 0) { throw "leakage check failed for MaxPerType=$size" }

    $metrics = Get-Content "$outDir\attribution_head_metrics.json" -Raw | ConvertFrom-Json
    $stats = Get-Content "$outDir\counterfactual_generation_stats.json" -Raw | ConvertFrom-Json
    $leak = Get-Content "$outDir\leakage_check.json" -Raw | ConvertFrom-Json

    foreach ($method in @("majority", "field_wise_nli", "logistic_regression_head", "mlp_head_wo_nli", "mlp_head_wo_evidence", "mlp_head_wo_graph", "attr_head_mlp")) {
        $res = $metrics.results.$method
        if ($null -eq $res) { continue }
        $rows += [pscustomobject]@{
            max_per_type = $size
            method = $method
            generated_rows = $stats.type_counts.PSObject.Properties.Value | Measure-Object -Sum | Select-Object -ExpandProperty Sum
            train_rows = $metrics.train_rows
            val_rows = $metrics.val_rows
            test_rows = $metrics.test_rows
            type_acc = [double]$res.mismatch_type_accuracy
            field_micro_f1 = [double]$res.conflict_field_micro_f1
            field_macro_f1 = [double]$res.conflict_field_macro_f1
            exact_match = [double]$res.exact_match_rate
            source_sample_id_leakage = [int]$leak.source_sample_id_leakage
            image_id_leakage = [int]$leak.image_id_leakage
            duplicate_caption_cross_split = [int]$leak.cross_split_duplicate_edited_caption
            used_transformers = -not [bool]$NoTransformers
            output_dir = $outDir
        }
    }
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputRoot "counterfactual_scaling_results.csv")
}

$csv = Join-Path $OutputRoot "counterfactual_scaling_results.csv"
if (-not $NoCopySummary) {
    Copy-Item $csv "outputs\counterfactual_scaling_results.csv" -Force
}
Write-Host "Scaling summary written:"
Write-Host "  $csv"
if (-not $NoCopySummary) { Write-Host "  outputs\counterfactual_scaling_results.csv" }
