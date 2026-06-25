param(
    [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
    [string]$Python = "python",
    [int]$MaxPerType = 80,
    [string]$NliModel = "facebook/bart-large-mnli",
    [int]$NliDevice = 0,
    [switch]$NoTransformers
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force outputs\counterfactual | Out-Null

function Run-Step {
    param([string]$Name, [string[]]$ArgList)
    Write-Host $Name
    & $Python @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name exit_code=$LASTEXITCODE"
    }
}

Run-Step "[1/6] build controlled counterfactual data" @(
    "scripts\data\build_controlled_counterfactuals.py",
    "--context-pairs", "outputs\cove_lite_context_pairs.jsonl",
    "--output-dir", "outputs\counterfactual",
    "--max-per-type", "$MaxPerType"
)

foreach ($split in @("train", "val", "test")) {
    Run-Step "[2/6] extract event tuples: $split" @(
        "scripts\event\extract_event_tuples_v2.py",
        "--input", "outputs\counterfactual\controlled_counterfactual_$split.jsonl",
        "--output", "outputs\counterfactual\controlled_counterfactual_${split}_events.jsonl",
        "--extractor", "enhanced"
    )
    $nliArgs = @(
        "scripts\attribution\run_field_nli_attribution_v2.py",
        "--input", "outputs\counterfactual\controlled_counterfactual_${split}_events.jsonl",
        "--output", "outputs\counterfactual\controlled_counterfactual_${split}_features.jsonl",
        "--model", $NliModel,
        "--device", "$NliDevice"
    )
    if ($NoTransformers) { $nliArgs += "--no-transformers" }
    Run-Step "[3/6] field-wise NLI features: $split" $nliArgs
}

Run-Step "[4/6] train attribution head" @(
    "scripts\train\train_attribution_head.py",
    "--train", "outputs\counterfactual\controlled_counterfactual_train_features.jsonl",
    "--val", "outputs\counterfactual\controlled_counterfactual_val_features.jsonl",
    "--test", "outputs\counterfactual\controlled_counterfactual_test_features.jsonl",
    "--model-out", "outputs\counterfactual\attribution_head_model.pkl",
    "--metrics-out", "outputs\counterfactual\attribution_head_metrics.json"
)

Run-Step "[5/6] collect v2 report tables" @(
    "scripts\collect_v2_report_tables.py",
    "--output", "outputs\report_tables_v2.md"
)

Write-Host "[6/6] done"
Write-Host "Key files:"
Write-Host "  outputs\counterfactual\counterfactual_generation_stats.json"
Write-Host "  outputs\counterfactual\controlled_counterfactual_train_features.jsonl"
Write-Host "  outputs\counterfactual\controlled_counterfactual_val_features.jsonl"
Write-Host "  outputs\counterfactual\controlled_counterfactual_test_features.jsonl"
Write-Host "  outputs\counterfactual\attribution_head_metrics.json"
Write-Host "  outputs\report_tables_v2.md"

