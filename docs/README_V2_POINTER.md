# Read this first: VDT-COVE-Attr v2

For the high-standard technical route and implementation details, read:

1. `docs/VDT_COVE_ATTR_V2_IMPLEMENTATION_ROUTER.md`
2. `docs/EXPERIMENT_TODO_AND_HANDOFF.md`
3. `docs/INNOVATION_POINTS_V2.md`
4. `requirements_vdt_cove_attr_v2.txt`

Run local experiments with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vdt_cove_attr_v2_experiments.ps1 `
  -ProjectRoot D:\MY_PROJECT\OOC\E3-VDT-OOC `
  -NewsClippingsDataDir D:\MY_PROJECT\OOC\datasets\NewsCLIPpings_repo\news_clippings\data `
  -VisualNewsMetadataDir E:\OOC_Datasets\VisualNews\articles_metadata `
  -Python python `
  -MaxRecords 500 `
  -EvalSampleN 80 `
  -NliModel facebook/bart-large-mnli
```

Rules:

- VDT remains the classifier.
- Current rule sidecar is a baseline only.
- Main route is true context + enhanced event extraction + field-wise NLI + evidence relevance.
- Weak labels are not gold labels.
- Report attribution quality only on manual gold annotation.
