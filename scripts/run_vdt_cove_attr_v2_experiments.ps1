param(
    [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
    [string]$NewsClippingsDataDir = "D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data",
    [string]$VisualNewsMetadataDir = "E:\OOC_Datasets\VisualNews\articles_metadata",
    [string]$Python = "python",
    [int]$MaxRecords = 500,
    [int]$EvalSampleN = 80,
    [string]$NliModel = "facebook/bart-large-mnli",
    [int]$NliDevice = -1,
    [switch]$NoTransformers
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force outputs | Out-Null
New-Item -ItemType Directory -Force examples | Out-Null

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Args
    )
    Write-Host $Name
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name exit_code=$LASTEXITCODE"
    }
}

# Remove stale files from previous failed runs, but never delete manual gold annotations.
$staleFiles = @(
    "outputs\cove_lite_context_pairs.jsonl",
    "outputs\cove_lite_context_pairs.jsonl.stats.json",
    "outputs\weak_attribution_labels.jsonl",
    "outputs\weak_attribution_labels.jsonl.stats.json",
    "outputs\event_tuples_v2.jsonl",
    "outputs\event_tuples_v2.jsonl.stats.json",
    "outputs\field_nli_attribution_v2.jsonl",
    "outputs\field_nli_attribution_v2.jsonl.stats.json",
    "outputs\attribution_eval_v2_metrics.json",
    "outputs\attribution_eval_v2_metrics.csv",
    "examples\attribution_eval_candidates.jsonl",
    "outputs\report_tables_v2.md"
)
foreach ($file in $staleFiles) {
    Remove-Item $file -Force -ErrorAction SilentlyContinue
}

Invoke-PythonStep "[0/7] input check" @(
    "scripts\diagnose_cove_lite_inputs.py",
    "--newsclippings-data-dir", $NewsClippingsDataDir,
    "--visualnews-metadata-dir", $VisualNewsMetadataDir,
    "--output", "outputs\input_check.json"
)

Invoke-PythonStep "[1/7] COVE-lite true context pairs" @(
    "scripts\context\build_cove_lite_context_pairs.py",
    "--newsclippings-data-dir", $NewsClippingsDataDir,
    "--visualnews-metadata-dir", $VisualNewsMetadataDir,
    "--output", "outputs\cove_lite_context_pairs.jsonl",
    "--max-records", "$MaxRecords"
)

Invoke-PythonStep "[2/7] v1 weak rule labels (baseline)" @(
    "scripts\labels\build_weak_attribution_from_context.py",
    "--input", "outputs\cove_lite_context_pairs.jsonl",
    "--output", "outputs\weak_attribution_labels.jsonl"
)

Invoke-PythonStep "[3/7] v2 event tuple extraction" @(
    "scripts\event\extract_event_tuples_v2.py",
    "--input", "outputs\weak_attribution_labels.jsonl",
    "--output", "outputs\event_tuples_v2.jsonl",
    "--extractor", "enhanced"
)

$nliArgs = @(
    "scripts\attribution\run_field_nli_attribution_v2.py",
    "--input", "outputs\event_tuples_v2.jsonl",
    "--output", "outputs\field_nli_attribution_v2.jsonl",
    "--model", $NliModel,
    "--device", "$NliDevice"
)
if ($NoTransformers) { $nliArgs += "--no-transformers" }
Invoke-PythonStep "[4/7] v2 field-wise NLI + evidence relevance + graph alignment" $nliArgs

Invoke-PythonStep "[5/7] build manual annotation candidates" @(
    "scripts\eval\build_attribution_eval_sample.py",
    "--context-pairs", "outputs\cove_lite_context_pairs.jsonl",
    "--weak-labels", "outputs\field_nli_attribution_v2.jsonl",
    "--output", "examples\attribution_eval_candidates.jsonl",
    "--n", "$EvalSampleN"
)

Write-Host "[6/7] evaluate if gold set exists"
if (Test-Path examples\attribution_eval_set.jsonl) {
    $goldOverlapText = & $Python scripts\eval\count_gold_pred_overlap.py `
        --gold examples\attribution_eval_set.jsonl `
        --pred outputs\field_nli_attribution_v2.jsonl
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: [6/7] gold overlap check exit_code=$LASTEXITCODE"
    }
    $goldOverlap = [int]($goldOverlapText | Select-Object -Last 1)
    if ($goldOverlap -gt 0) {
        Invoke-PythonStep "[6/7] evaluate against examples\attribution_eval_set.jsonl overlap=$goldOverlap" @(
            "scripts\eval\evaluate_attribution_v2.py",
            "--gold", "examples\attribution_eval_set.jsonl",
            "--pred", "outputs\field_nli_attribution_v2.jsonl",
            "--output", "outputs\attribution_eval_v2_metrics.json"
        )
    } else {
        Write-Host "Gold file exists but has no overlapping sample_id with this run; skip evaluation to avoid stale zero metrics."
        Write-Host "After manual annotation, copy examples\attribution_eval_candidates.jsonl to examples\attribution_eval_set.jsonl or keep the same sampling seed."
    }
} else {
    Write-Host "Gold file missing."
    Write-Host "Copy examples\attribution_eval_candidates.jsonl to examples\attribution_eval_set.jsonl and fill gold_mismatch_type / gold_conflict_fields."
}

Invoke-PythonStep "[7/7] collect tables" @(
    "scripts\collect_v2_report_tables.py",
    "--output", "outputs\report_tables_v2.md"
)

Write-Host "Done. Key files:"
Write-Host "  outputs\input_check.json"
Write-Host "  outputs\cove_lite_context_pairs.jsonl.stats.json"
Write-Host "  outputs\event_tuples_v2.jsonl.stats.json"
Write-Host "  outputs\field_nli_attribution_v2.jsonl.stats.json"
Write-Host "  examples\attribution_eval_candidates.jsonl"
Write-Host "  outputs\report_tables_v2.md"
