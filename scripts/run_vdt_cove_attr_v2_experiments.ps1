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

# Remove stale files from previous failed runs, but never delete manual gold annotations.
$stale = @(
  "outputs\cove_lite_context_pairs.jsonl",
  "outputs\cove_lite_context_pairs.jsonl.stats.json",
  "outputs\weak_attribution_labels.jsonl",
  "outputs\weak_attribution_labels.jsonl.stats.json",
  "outputs\event_tuples_v2.jsonl",
  "outputs\event_tuples_v2.jsonl.stats.json",
  "outputs\field_nli_attribution_v2.jsonl",
  "outputs\field_nli_attribution_v2.jsonl.stats.json",
  "examples\attribution_eval_candidates.jsonl",
  "outputs\report_tables_v2.md"
)
foreach ($f in $stale) {
  if (Test-Path $f) { Remove-Item $f -Force }
}

Write-Host "[0/7] input check"
& $Python scripts\diagnose_cove_lite_inputs.py `
  --newsclippings-data-dir $NewsClippingsDataDir `
  --visualnews-metadata-dir $VisualNewsMetadataDir `
  --output outputs\input_check.json

Write-Host "[1/7] COVE-lite true context pairs"
& $Python scripts\context\build_cove_lite_context_pairs.py `
  --newsclippings-data-dir $NewsClippingsDataDir `
  --visualnews-metadata-dir $VisualNewsMetadataDir `
  --output outputs\cove_lite_context_pairs.jsonl `
  --max-records $MaxRecords

Write-Host "[2/7] v1 weak rule labels (baseline)"
& $Python scripts\labels\build_weak_attribution_from_context.py `
  --input outputs\cove_lite_context_pairs.jsonl `
  --output outputs\weak_attribution_labels.jsonl

Write-Host "[3/7] v2 event tuple extraction"
& $Python scripts\event\extract_event_tuples_v2.py `
  --input outputs\cove_lite_context_pairs.jsonl `
  --output outputs\event_tuples_v2.jsonl `
  --extractor enhanced

Write-Host "[4/7] v2 field-wise NLI + evidence relevance + graph alignment"
$nliArgs = @(
  "scripts\attribution\run_field_nli_attribution_v2.py",
  "--input", "outputs\event_tuples_v2.jsonl",
  "--output", "outputs\field_nli_attribution_v2.jsonl",
  "--model", $NliModel,
  "--device", "$NliDevice"
)
if ($NoTransformers) { $nliArgs += "--no-transformers" }
& $Python @nliArgs

Write-Host "[5/7] build manual annotation candidates"
& $Python scripts\eval\build_attribution_eval_sample.py `
  --context-pairs outputs\cove_lite_context_pairs.jsonl `
  --weak-labels outputs\field_nli_attribution_v2.jsonl `
  --output examples\attribution_eval_candidates.jsonl `
  --n $EvalSampleN

Write-Host "[6/7] evaluate if gold set exists and predictions are non-empty"
if ((Test-Path examples\attribution_eval_set.jsonl) -and (Test-Path outputs\field_nli_attribution_v2.jsonl) -and ((Get-Item outputs\field_nli_attribution_v2.jsonl).Length -gt 0)) {
  & $Python scripts\eval\evaluate_attribution_v2.py `
    --gold examples\attribution_eval_set.jsonl `
    --pred outputs\field_nli_attribution_v2.jsonl `
    --output outputs\attribution_eval_v2_metrics.json
} else {
  Write-Host "Gold file missing or predictions empty."
  Write-Host "Copy examples\attribution_eval_candidates.jsonl to examples\attribution_eval_set.jsonl and fill gold_mismatch_type / gold_conflict_fields."
}

Write-Host "[7/7] collect tables"
& $Python scripts\collect_v2_report_tables.py `
  --output outputs\report_tables_v2.md

Write-Host "Done. Key files:"
Write-Host "  outputs\input_check.json"
Write-Host "  outputs\cove_lite_context_pairs.jsonl.stats.json"
Write-Host "  outputs\event_tuples_v2.jsonl.stats.json"
Write-Host "  outputs\field_nli_attribution_v2.jsonl.stats.json"
Write-Host "  examples\attribution_eval_candidates.jsonl"
Write-Host "  outputs\report_tables_v2.md"
