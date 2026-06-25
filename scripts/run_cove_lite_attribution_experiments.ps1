param(
  [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
  [string]$NewsClippingsDataDir = "D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data",
  [string]$VisualNewsMetadataDir = "E:\OOC_Datasets\VisualNews\metadata",
  [string]$Python = "python",
  [int]$MaxRecords = 0,
  [int]$EvalSampleN = 120
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force outputs | Out-Null
New-Item -ItemType Directory -Force examples | Out-Null

Write-Host "[1/4] Build COVE-lite context pairs"
$context = Join-Path $ProjectRoot "outputs\cove_lite_context_pairs.jsonl"
$args1 = @(
  "scripts\context\build_cove_lite_context_pairs.py",
  "--newsclippings-data-dir", $NewsClippingsDataDir,
  "--visualnews-metadata-dir", $VisualNewsMetadataDir,
  "--output", $context
)
if ($MaxRecords -gt 0) { $args1 += @("--max-records", "$MaxRecords") }
& $Python @args1

Write-Host "[2/4] Build weak attribution labels"
$weak = Join-Path $ProjectRoot "outputs\weak_attribution_labels.jsonl"
& $Python "scripts\labels\build_weak_attribution_from_context.py" `
  --input $context `
  --output $weak

Write-Host "[3/4] Build manual annotation candidates"
$candidates = Join-Path $ProjectRoot "examples\attribution_eval_candidates.jsonl"
& $Python "scripts\eval\build_attribution_eval_sample.py" `
  --context-pairs $context `
  --weak-labels $weak `
  --output $candidates `
  --n $EvalSampleN

Write-Host "[4/4] If gold file exists, evaluate attribution baselines"
$gold = Join-Path $ProjectRoot "examples\attribution_eval_set.jsonl"
if (Test-Path $gold) {
  & $Python "scripts\eval\run_attribution_baselines.py" `
    --gold $gold `
    --weak-labels $weak `
    --output "outputs\attribution_eval_metrics.json"
} else {
  Write-Host "Gold file not found: $gold"
  Write-Host "Next step: copy examples\attribution_eval_candidates.jsonl to examples\attribution_eval_set.jsonl and fill gold_mismatch_type / gold_conflict_fields."
}

Write-Host "Done. Generated files:"
Write-Host "  $context"
Write-Host "  $weak"
Write-Host "  $candidates"
