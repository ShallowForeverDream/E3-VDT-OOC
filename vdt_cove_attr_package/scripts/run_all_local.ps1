param(
  [string]$ProjectRoot = "D:\MY_PROJECT\OOC\E3-VDT-OOC",
  [string]$NewsClippingsDataDir = "D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data",
  [string]$VisualNewsMetadataDir = "E:\OOC_Datasets\VisualNews\articles_metadata",
  [string]$Python = "python",
  [int]$MaxRecords = 500,
  [int]$EvalSampleN = 80,
  [string]$NliModel = "facebook/bart-large-mnli",
  [switch]$NoNli
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force outputs, examples | Out-Null

function RunStep($name, $cmd) {
  Write-Host "`n=== $name ===" -ForegroundColor Cyan
  Write-Host $cmd
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $name exit=$LASTEXITCODE" }
}

$maxArg = ""
if ($MaxRecords -gt 0) { $maxArg = "--max-records $MaxRecords" }

RunStep "00 check inputs" "`"$Python`" `"$PackageRoot\scripts\00_check_inputs.py`" --news-dir `"$NewsClippingsDataDir`" --visual-dir `"$VisualNewsMetadataDir`" --out outputs\input_check.json"

RunStep "01 build COVE-lite context pairs" "`"$Python`" `"$PackageRoot\scripts\01_build_context_pairs.py`" --news-dir `"$NewsClippingsDataDir`" --visual-dir `"$VisualNewsMetadataDir`" --output outputs\cove_lite_context_pairs.jsonl $maxArg"

RunStep "02 extract event tuples" "`"$Python`" `"$PackageRoot\scripts\02_extract_events.py`" --input outputs\cove_lite_context_pairs.jsonl --output outputs\event_tuples.jsonl --extractor rule"

$nliArgs = "--nli-model `"$NliModel`""
if ($NoNli) { $nliArgs = "--no-nli" }
RunStep "03 field-wise NLI attribution" "`"$Python`" `"$PackageRoot\scripts\03_field_nli_attribution.py`" --input outputs\event_tuples.jsonl --output outputs\field_nli_attribution.jsonl $nliArgs --device 0"

RunStep "04 build annotation candidates" "`"$Python`" `"$PackageRoot\scripts\04_build_annotation_candidates.py`" --input outputs\field_nli_attribution.jsonl --output examples\attribution_eval_candidates.jsonl --n $EvalSampleN"

if (Test-Path "examples\attribution_eval_set.jsonl") {
  RunStep "05 evaluate attribution" "`"$Python`" `"$PackageRoot\scripts\05_eval_attribution.py`" --gold examples\attribution_eval_set.jsonl --pred outputs\field_nli_attribution.jsonl --output outputs\attribution_eval_metrics.json"
} else {
  Write-Host "Gold file not found: examples\attribution_eval_set.jsonl" -ForegroundColor Yellow
  Write-Host "Next: copy examples\attribution_eval_candidates.jsonl to examples\attribution_eval_set.jsonl and fill gold labels." -ForegroundColor Yellow
}

RunStep "06 collect report tables" "`"$Python`" `"$PackageRoot\scripts\06_collect_tables.py`" --output-dir outputs --out outputs\report_tables.md"

Write-Host "`nDone. Check outputs\report_tables.md and examples\attribution_eval_candidates.jsonl" -ForegroundColor Green
