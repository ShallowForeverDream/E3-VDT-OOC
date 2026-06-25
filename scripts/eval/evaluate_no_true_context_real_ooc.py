from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.infer_vdt_cf_attr import (  # noqa: E402
    FIELDS,
    TYPE_TO_FIELDS,
    postprocess_prediction,
    predict_with_model,
    prompt_rule,
)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_type(t: Any) -> str:
    t = str(t or "").strip()
    if t == "none":
        return "benign illustrative image"
    if t == "uncertain / evidence insufficient":
        return "uncertain / insufficient visual evidence"
    return t


def gold_fields(row: Dict[str, Any]) -> Set[str]:
    fields = {f for f in FIELDS if str(row.get(f"gold_field_{f}") or "0").strip() in {"1", "1.0", "true", "True"}}
    if not fields:
        fields = set(TYPE_TO_FIELDS.get(norm_type(row.get("gold_mismatch_type")), []))
    return fields


def predict_row(row: Dict[str, Any], model_path: Path) -> Tuple[str, Set[str], float, str]:
    pred = predict_with_model(row, model_path)
    if pred is None:
        mismatch_type, fields, confidence = prompt_rule(row)
        source = "field_prompt_grounding_rule_fallback"
    else:
        mismatch_type, fields, confidence, _ = pred
        source = "no_true_context_attr_head"
    mismatch_type, fields, _, _ = postprocess_prediction(
        mismatch_type=mismatch_type,
        conflict_fields=set(fields),
        vdt_label=str(row.get("vdt_label") or "OOC"),
        feat=row,
    )
    return norm_type(mismatch_type), set(fields), float(confidence), source


def score(rows: Sequence[Dict[str, Any]], model_path: Path) -> Dict[str, Any]:
    type_ok = exact = 0
    tp = fp = fn = 0
    per_field = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    confusion: Counter[str] = Counter()
    pred_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        gt = norm_type(row.get("gold_mismatch_type"))
        gf = gold_fields(row)
        pt, pf, _, src = predict_row(row, model_path)
        pred_type_counts[pt] += 1
        source_counts[src] += 1
        type_ok += int(gt == pt)
        exact += int(gf == pf)
        confusion[f"{gt} -> {pt}"] += 1
        for f in FIELDS:
            ing, inp = f in gf, f in pf
            if ing and inp:
                tp += 1
                per_field[f]["tp"] += 1
            elif inp and not ing:
                fp += 1
                per_field[f]["fp"] += 1
            elif ing and not inp:
                fn += 1
                per_field[f]["fn"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_vals = []
    per_field_f1 = {}
    for f, c in per_field.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_field_f1[f] = f1
        macro_vals.append(f1)
    n = len(rows)
    return {
        "model_path": str(model_path),
        "n": n,
        "mismatch_type_accuracy": type_ok / n if n else 0.0,
        "conflict_field_micro_precision": precision,
        "conflict_field_micro_recall": recall,
        "conflict_field_micro_f1": micro_f1,
        "conflict_field_macro_f1": sum(macro_vals) / len(macro_vals) if macro_vals else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "pred_type_counts": dict(pred_type_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "per_field_f1": per_field_f1,
        "type_confusion_top": dict(confusion.most_common(20)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate no-true-context attribution heads on the 100-row real OOC manual gold set.")
    ap.add_argument("--features", default="outputs/real_ooc_manual_no_true_context_features.csv")
    ap.add_argument("--model", action="append", required=True, help="Path to no_true_context_attr_head.pkl; can be repeated.")
    ap.add_argument("--output", default="outputs/real_ooc_no_true_context_eval_metrics.json")
    args = ap.parse_args()

    rows = read_csv(Path(args.features))
    results = {Path(m).parent.name or Path(m).name: score(rows, Path(m)) for m in args.model}
    payload = {
        "features": args.features,
        "records": len(rows),
        "models": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n", "mismatch_type_accuracy", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate"],
        )
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({"model": name, **{k: res.get(k) for k in writer.fieldnames if k != "model"}})
    print(json.dumps({"output": str(out), "csv": str(csv_path), "models": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
