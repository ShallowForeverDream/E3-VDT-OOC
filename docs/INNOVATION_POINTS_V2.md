# Innovation Points: VDT-COVE-Attr v2

## Current project stance

This project does **not** claim that E3-VDT replaces VDT or improves VDT's main OOC classification accuracy.

The validated baseline is:

- VDT strict BLIP-2/GaussianBlur on `bbc,guardian`: F1 0.7353 / Acc 0.7383 / AUC 0.7398.
- VDT strict BLIP-2/GaussianBlur on `usa_today,washington_post`: F1 0.8032 / Acc 0.8032 / AUC 0.8028.

The research problem we address is:

> VDT predicts OOC / Non-OOC, but it does not explain which event field is wrong.

## What is only a baseline

The original rule-based event extractor and string-similarity mismatch detector are retained only as a **Rule Sidecar Baseline**. They are not the final method.

## Final route

The final route is **VDT-COVE-Attr v2**:

```text
VDT baseline
+ COVE-lite true context
+ enhanced event extraction
+ evidence relevance / sufficiency
+ field-wise NLI contradiction detection
+ lightweight graph alignment
+ AMG-style manual attribution evaluation
```

## Core innovations

1. **Accuracy-preserving VDT attribution sidecar**: VDT remains the source of the final OOC / Non-OOC label; the sidecar adds mismatch type and conflict fields.
2. **COVE-lite true-context grounding**: VisualNews original caption/title/article metadata becomes true image context for NewsCLIPpings images.
3. **Enhanced event extraction**: rule fallback + optional NER + time/location normalization + OpenIE-like triples + LLM JSON extension point.
4. **Field-wise NLI contradiction attribution**: fields become conflicts only when NLI marks the corresponding hypothesis as contradiction.
5. **Evidence relevance / sufficiency**: insufficient context is marked as evidence insufficient instead of forced into a wrong mismatch type.
6. **Lightweight graph alignment**: subject-predicate-object triples are aligned to catch relation/object conflicts.

## Required validation

The attribution module is only valid if supported by experiments:

1. COVE-lite coverage table.
2. Event extraction coverage and optional field F1.
3. Attribution comparison on manual gold set.
4. Evidence relevance ablation.
5. Hard negative field-F1.

The final report must not call weak labels ground truth. It must separate weak automatic labels from manual gold labels.
