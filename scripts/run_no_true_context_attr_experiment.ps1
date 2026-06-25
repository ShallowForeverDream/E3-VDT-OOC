param(
    [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
    [string]$Python = "python",
    [int]$MaxPerType = 80,
    [string]$ContextPairs = "outputs\cove_lite_context_pairs.jsonl",
    [string]$OutputDir = "outputs\no_true_context_attr",
    [string]$OriginDataJson = "E:\OOC_Datasets\VisualNews\origin\data.json",
    [string]$OriginTar = "E:\OOC_Datasets\VisualNews\origin.tar",
    [string]$TarIndex = "D:\MY_PROJECT\OOC\datasets\visualnews_train_test_tar_index.json",
    [string]$ClipModel = "openai/clip-vit-base-patch32",
    [string]$Device = "cuda",
    [int]$BatchSize = 16,
    [switch]$ReuseCounterfactual,
    [switch]$NoClip,
    [switch]$IncludeDifferentEvent,
    [int]$MaxDifferentEvent = 0,
    [double]$DifferentEventMaxSimilarity = 0.65,
    [double]$DifferentEventMaxTokenJaccard = 0.08,
    [string]$DifferentEventExcludeGold = "examples\real_ooc_attribution_eval_set.jsonl"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $OutputDir | Out-Null

function Run-Step {
    param([string]$Name, [string[]]$ArgList)
    Write-Host $Name
    & $Python @ArgList
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name exit_code=$LASTEXITCODE" }
}

if (-not $ReuseCounterfactual) {
    $buildArgs = @(
        "scripts\data\build_controlled_counterfactuals.py",
        "--context-pairs", $ContextPairs,
        "--output-dir", $OutputDir,
        "--max-per-type", "$MaxPerType"
    )
    if ($IncludeDifferentEvent) {
        $effectiveMaxDifferent = $MaxDifferentEvent
        if ($effectiveMaxDifferent -le 0) { $effectiveMaxDifferent = $MaxPerType }
        $buildArgs += @(
            "--include-original-ooc-different-event",
            "--max-different-event", "$effectiveMaxDifferent",
            "--different-event-max-similarity", "$DifferentEventMaxSimilarity",
            "--different-event-max-token-jaccard", "$DifferentEventMaxTokenJaccard"
        )
        if (Test-Path $DifferentEventExcludeGold) {
            $buildArgs += @("--exclude-different-event-gold", $DifferentEventExcludeGold)
        }
    }
    Run-Step "[1/5] build controlled counterfactual data with group split" $buildArgs
} else {
    Write-Host "[1/5] reuse existing counterfactual jsonl in $OutputDir"
}

Run-Step "[2/5] leakage check" @(
    "scripts\eval\check_counterfactual_leakage.py",
    "--train", "$OutputDir\controlled_counterfactual_train.jsonl",
    "--val", "$OutputDir\controlled_counterfactual_val.jsonl",
    "--test", "$OutputDir\controlled_counterfactual_test.jsonl",
    "--output", "$OutputDir\leakage_check.json",
    "--fail-on-leak"
)

foreach ($split in @("train", "val", "test")) {
    $args = @(
        "scripts\features\build_image_caption_attribution_features.py",
        "--input", "$OutputDir\controlled_counterfactual_$split.jsonl",
        "--output", "$OutputDir\image_caption_features_$split.csv",
        "--origin-data-json", $OriginDataJson,
        "--origin-tar", $OriginTar,
        "--tar-index", $TarIndex,
        "--clip-model", $ClipModel,
        "--device", $Device,
        "--batch-size", "$BatchSize"
    )
    if ($NoClip) { $args += "--no-clip" }
    Run-Step "[3/5] build no-true-context features: $split" $args
}

Run-Step "[4/5] train no-true-context attribution head" @(
    "scripts\train\train_no_true_context_attribution_head.py",
    "--train", "$OutputDir\image_caption_features_train.csv",
    "--val", "$OutputDir\image_caption_features_val.csv",
    "--test", "$OutputDir\image_caption_features_test.csv",
    "--model-out", "$OutputDir\no_true_context_attr_head.pkl",
    "--metrics-out", "$OutputDir\no_true_context_attr_metrics.json"
)

Run-Step "[5/5] collect report tables" @(
    "scripts\collect_v2_report_tables.py",
    "--output", "outputs\report_tables_v2.md"
)

Write-Host "Done. Key files:"
Write-Host "  $OutputDir\image_caption_features_train.csv"
Write-Host "  $OutputDir\image_caption_features_val.csv"
Write-Host "  $OutputDir\image_caption_features_test.csv"
Write-Host "  $OutputDir\no_true_context_attr_metrics.json"
Write-Host "  $OutputDir\leakage_check.json"

