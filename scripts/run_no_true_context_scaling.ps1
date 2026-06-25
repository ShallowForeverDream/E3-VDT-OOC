param(
    [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
    [string]$Python = "python",
    [string[]]$Sizes = @("80", "200", "1000"),
    [string]$ContextPairs = "outputs\cove_lite_context_pairs.jsonl",
    [string]$NewsClippingsDataDir = "",
    [string]$VisualNewsMetadataDir = "",
    [int]$ContextMaxRecords = 3000,
    [switch]$RebuildContextPairs,
    [string]$OutputRoot = "outputs\no_true_context_scaling",
    [string]$CsvOut = "outputs\no_true_context_scaling_results.csv",
    [string]$OriginDataJson = "E:\OOC_Datasets\VisualNews\origin\data.json",
    [string]$OriginTar = "E:\OOC_Datasets\VisualNews\origin.tar",
    [string]$TarIndex = "D:\MY_PROJECT\OOC\datasets\visualnews_train_test_tar_index.json",
    [string]$ClipModel = "openai/clip-vit-base-patch32",
    [string]$Device = "cuda",
    [int]$BatchSize = 16,
    [switch]$NoClip
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

if ($RebuildContextPairs -or -not (Test-Path $ContextPairs)) {
    if ($NewsClippingsDataDir -and $VisualNewsMetadataDir) {
        Write-Host "===== building context pairs: $ContextPairs max_records=$ContextMaxRecords ====="
        & $Python "scripts\context\build_cove_lite_context_pairs.py" `
            "--newsclippings-data-dir" $NewsClippingsDataDir `
            "--visualnews-metadata-dir" $VisualNewsMetadataDir `
            "--output" $ContextPairs `
            "--max-records" "$ContextMaxRecords"
        if ($LASTEXITCODE -ne 0) {
            throw "build_cove_lite_context_pairs failed exit_code=$LASTEXITCODE"
        }
    } else {
        throw "ContextPairs not found: $ContextPairs. Provide -NewsClippingsDataDir and -VisualNewsMetadataDir, or build context pairs first."
    }
}

function Read-JsonOrEmpty {
    param([string]$Path)
    if (Test-Path $Path) {
        return Get-Content $Path -Raw | ConvertFrom-Json
    }
    return $null
}

function Get-Prop {
    param($Obj, [string]$Name, $Default = "")
    if ($null -eq $Obj) { return $Default }
    $p = $Obj.PSObject.Properties[$Name]
    if ($null -eq $p) { return $Default }
    return $p.Value
}

$runner = Join-Path $ProjectRoot "scripts\run_no_true_context_attr_experiment.ps1"
$rows = @()
$normalizedSizes = @()
foreach ($entry in $Sizes) {
    foreach ($part in (("$entry") -split ",")) {
        $p = $part.Trim()
        if ($p) { $normalizedSizes += [int]$p }
    }
}

foreach ($size in $normalizedSizes) {
    $outDir = Join-Path $OutputRoot ("max_{0}" -f $size)
    Write-Host "===== no-true-context scaling MaxPerType=$size output=$outDir ====="

    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $runner,
        "-ProjectRoot", $ProjectRoot,
        "-Python", $Python,
        "-MaxPerType", "$size",
        "-ContextPairs", $ContextPairs,
        "-OutputDir", $outDir,
        "-OriginDataJson", $OriginDataJson,
        "-OriginTar", $OriginTar,
        "-TarIndex", $TarIndex,
        "-ClipModel", $ClipModel,
        "-Device", $Device,
        "-BatchSize", "$BatchSize"
    )
    if ($NoClip) { $args += "-NoClip" }
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Scaling run failed: MaxPerType=$size exit_code=$LASTEXITCODE"
    }

    $metrics = Read-JsonOrEmpty (Join-Path $outDir "no_true_context_attr_metrics.json")
    $leak = Read-JsonOrEmpty (Join-Path $outDir "leakage_check.json")
    $gen = Read-JsonOrEmpty (Join-Path $outDir "counterfactual_generation_stats.json")

    $split = Get-Prop $metrics "split_counts" $null
    $typeCounts = Get-Prop $gen "type_counts" $null
    $timeSummary = Get-Prop $gen "time_swap_summary" $null
    $results = Get-Prop $metrics "results" $null
    if ($null -eq $results) { continue }

    foreach ($methodProp in $results.PSObject.Properties) {
        $name = $methodProp.Name
        $res = $methodProp.Value
        $trainRows = Get-Prop $metrics "train_rows" (Get-Prop $split "train" 0)
        $valRows = Get-Prop $metrics "val_rows" (Get-Prop $split "val" 0)
        $testRows = Get-Prop $metrics "test_rows" (Get-Prop $split "test" 0)
        $rows += [pscustomobject]@{
            max_per_type = $size
            output_dir = $outDir
            method = $name
            train_rows = $trainRows
            val_rows = $valRows
            test_rows = $testRows
            n = Get-Prop $res "n" 0
            type_acc = Get-Prop $res "mismatch_type_accuracy" 0
            field_micro_f1 = Get-Prop $res "conflict_field_micro_f1" 0
            field_macro_f1 = Get-Prop $res "conflict_field_macro_f1" 0
            exact_match = Get-Prop $res "exact_match_rate" 0
            source_sample_id_leakage = Get-Prop $leak "source_sample_id_leakage" 0
            image_id_leakage = Get-Prop $leak "image_id_leakage" 0
            text_id_leakage = Get-Prop $leak "text_id_leakage" 0
            cross_split_duplicate_edited_caption = Get-Prop $leak "cross_split_duplicate_edited_caption" 0
            none_count = Get-Prop $typeCounts "none" 0
            location_swap_count = Get-Prop $typeCounts "location_swap" 0
            time_swap_count = Get-Prop $typeCounts "time_swap" 0
            entity_swap_count = Get-Prop $typeCounts "entity_swap" 0
            time_swap_generated = Get-Prop $timeSummary "generated" 0
            time_swap_skipped_no_span = Get-Prop $timeSummary "skipped_no_span" 0
            uses_true_context_at_inference = Get-Prop $metrics "uses_true_context_at_inference" $false
        }
    }
}

$csvPath = Join-Path $ProjectRoot $CsvOut
New-Item -ItemType Directory -Force (Split-Path $csvPath -Parent) | Out-Null
$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

& $Python "scripts\collect_v2_report_tables.py" "--output" "outputs\report_tables_v2.md"
if ($LASTEXITCODE -ne 0) { throw "collect_v2_report_tables failed" }

Write-Host "Done: $CsvOut"
