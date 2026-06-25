from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

FIELDS = ["entity", "location", "time", "event_type", "relation", "context_omission", "evidence_insufficient"]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key_of(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("image_id") or row.get("current_caption") or "")


def field_set(xs: Any) -> Set[str]:
    if not xs:
        return set()
    if isinstance(xs, str):
        xs = [xs]
    return {str(x).strip() for x in xs if str(x).strip()}


def micro_f1(golds: List[Set[str]], preds: List[Set[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1}


def macro_f1(golds: List[Set[str]], preds: List[Set[str]]) -> float:
    vals = []
    for field in FIELDS:
        gbin = [{field} if field in g else set() for g in golds]
        pbin = [{field} if field in p else set() for p in preds]
        vals.append(micro_f1(gbin, pbin)["micro_f1"])
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], pred_type_key: str, pred_fields_key: str) -> Dict[str, Any]:
    pred_by_key = {key_of(r): r for r in pred_rows}
    matched = []
    missing = 0
    for g in gold_rows:
        k = key_of(g)
        if k in pred_by_key:
            matched.append((g, pred_by_key[k]))
        else:
            missing += 1
    type_correct = 0
    gold_fields: List[Set[str]] = []
    pred_fields: List[Set[str]] = []
    exact = 0
    type_confusion = Counter()
    for g, p in matched:
        gt = str(g.get("gold_mismatch_type", "")).strip()
        pt = str(p.get(pred_type_key, "")).strip()
        if gt == pt:
            type_correct += 1
        type_confusion[f"{gt} -> {pt}"] += 1
        gf = field_set(g.get("gold_conflict_fields"))
        pf = field_set(p.get(pred_fields_key))
        gold_fields.append(gf)
        pred_fields.append(pf)
        if gf == pf:
            exact += 1
    n = len(matched)
    m = micro_f1(gold_fields, pred_fields) if n else {"micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
    return {
        "matched": n,
        "missing_predictions": missing,
        "mismatch_type_accuracy": type_correct / n if n else 0.0,
        "conflict_field_micro_precision": m["micro_precision"],
        "conflict_field_micro_recall": m["micro_recall"],
        "conflict_field_micro_f1": m["micro_f1"],
        "conflict_field_macro_f1": macro_f1(gold_fields, pred_fields) if n else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "type_confusion_top": dict(type_confusion.most_common(20)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate VDT-COVE-Attr v2 predictions against manual gold attribution labels.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    gold = [r for r in load_jsonl(args.gold) if r.get("annotation_status", "done") == "done" or r.get("gold_mismatch_type")]
    pred = load_jsonl(args.pred)
    methods = {
        "v2_field_nli_evidence_graph": ("v2_mismatch_type", "v2_conflict_fields"),
        "weak_rule_sidecar": ("weak_mismatch_type", "weak_conflict_fields"),
    }
    results = {name: evaluate(gold, pred, a, b) for name, (a, b) in methods.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "matched", "mismatch_type_accuracy", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate", "missing_predictions"])
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({"method": name, **{k: res.get(k) for k in writer.fieldnames if k != "method"}})
    print(json.dumps({"output": str(out_path), "csv": str(csv_path), "methods": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
