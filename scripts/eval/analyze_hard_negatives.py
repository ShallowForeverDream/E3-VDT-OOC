from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_real_ooc_attribution import (  # noqa: E402
    attr_head_predictions,
    field_set,
    gold_rows,
    key_of,
    prediction_file_predictions,
    read_jsonl,
    rule_sidecar_predictions,
    score,
)


def subset_name(row: Dict[str, Any], pred: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    ev = pred.get("evidence_relevance") or {}
    overlaps = ev.get("field_overlaps") or {}
    fields = field_set(pred.get("v2_conflict_fields"))
    sim = float(ev.get("text_similarity") or 0.0)
    if sim >= 0.75:
        names.append("high-sim OOC")
    if overlaps.get("entity", 0.0) >= 0.5 and "time" in fields:
        names.append("same-person diff-time")
    if overlaps.get("event_type", 0.0) >= 0.5 and "location" in fields:
        names.append("same-topic diff-location")
    if overlaps.get("location", 0.0) >= 0.5 and ("event_type" in fields or "entity" in fields):
        names.append("same-location different-event")
    if not names:
        names.append("other")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate real OOC attribution methods on hard-negative subsets.")
    ap.add_argument("--gold", default="examples/real_ooc_attribution_eval_set.jsonl")
    ap.add_argument("--predictions", default="outputs/field_nli_attribution_v2.jsonl")
    ap.add_argument("--attr-head-model", default="outputs/counterfactual/attribution_head_model.pkl")
    ap.add_argument("--output", default="outputs/hard_negative_attribution_analysis.json")
    args = ap.parse_args()

    gold = gold_rows(read_jsonl(Path(args.gold)))
    pred_rows = read_jsonl(Path(args.predictions))
    pred_by_key = {key_of(r): r for r in pred_rows}
    subsets: Dict[str, List[Dict[str, Any]]] = {}
    for row in gold:
        pred = pred_by_key.get(key_of(row), {})
        for name in subset_name(row, pred):
            subsets.setdefault(name, []).append(row)

    method_preds = {
        "rule_sidecar": rule_sidecar_predictions(gold),
        "field_wise_nli": prediction_file_predictions(pred_rows, "v2_mismatch_type", "v2_conflict_fields"),
        "counterfactual_trained_attr_head": attr_head_predictions(pred_rows, Path(args.attr_head_model)),
    }
    result: Dict[str, Any] = {}
    for subset, rows in subsets.items():
        result[subset] = {"n": len(rows), "methods": {name: score(rows, preds) for name, preds in method_preds.items()}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "subsets": {k: v["n"] for k, v in result.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

