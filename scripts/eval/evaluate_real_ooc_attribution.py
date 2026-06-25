from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline  # noqa: E402
from scripts.train.train_attribution_head import FIELDS, feature_vector  # noqa: E402


TYPE_TO_FIELDS = {
    "benign illustrative image": [],
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "different-event mismatch": ["entity", "location", "event_type"],
    "global/uncontrolled mismatch": ["different_event"],
    "evidence insufficient": ["evidence_insufficient"],
    "uncertain / evidence insufficient": ["evidence_insufficient"],
}
ALL_FIELDS = list(dict.fromkeys(FIELDS + ["different_event"]))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key_of(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("image_id") or row.get("current_caption") or "").strip()


def field_set(xs: Any) -> Set[str]:
    if isinstance(xs, str):
        parts = [x.strip() for x in xs.replace(";", ",").split(",")]
    else:
        parts = list(xs or [])
    return {str(x).strip() for x in parts if str(x).strip()}


def gold_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if str(r.get("annotation_status", "done")).strip() in {"done", ""}
        and str(r.get("gold_mismatch_type") or "").strip()
    ]


def micro_f1(golds: Sequence[Set[str]], preds: Sequence[Set[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(golds: Sequence[Set[str]], preds: Sequence[Set[str]]) -> float:
    vals = []
    for field in ALL_FIELDS:
        gb = [{field} if field in g else set() for g in golds]
        pb = [{field} if field in p else set() for p in preds]
        vals.append(micro_f1(gb, pb)["f1"])
    return sum(vals) / len(vals) if vals else 0.0


def normalize_type(t: str) -> str:
    t = str(t or "").strip()
    if t == "global/uncontrolled mismatch":
        return "different-event mismatch"
    if t == "evidence insufficient":
        return "uncertain / evidence insufficient"
    return t


def type_fields(t: str) -> Set[str]:
    return set(TYPE_TO_FIELDS.get(t, []))


def score(gold: Sequence[Dict[str, Any]], preds: Dict[str, Tuple[str, Set[str]]]) -> Dict[str, Any]:
    matched = []
    missing = 0
    for g in gold:
        k = key_of(g)
        if k in preds:
            matched.append((g, preds[k]))
        else:
            missing += 1
    type_correct = 0
    exact = 0
    gold_fields: List[Set[str]] = []
    pred_fields: List[Set[str]] = []
    confusion = Counter()
    for g, (pt, pf) in matched:
        gt = normalize_type(str(g.get("gold_mismatch_type") or ""))
        pt = normalize_type(pt)
        gf = field_set(g.get("gold_conflict_fields")) or type_fields(gt)
        if gt == pt:
            type_correct += 1
        if gf == pf:
            exact += 1
        confusion[f"{gt} -> {pt}"] += 1
        gold_fields.append(gf)
        pred_fields.append(pf)
    n = len(matched)
    m = micro_f1(gold_fields, pred_fields) if n else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "matched": n,
        "missing_predictions": missing,
        "mismatch_type_accuracy": type_correct / n if n else 0.0,
        "conflict_field_micro_precision": m["precision"],
        "conflict_field_micro_recall": m["recall"],
        "conflict_field_micro_f1": m["f1"],
        "conflict_field_macro_f1": macro_f1(gold_fields, pred_fields) if n else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "type_confusion_top": dict(confusion.most_common(20)),
    }


def rule_sidecar_predictions(gold: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[str, Set[str]]]:
    pipe = VDTCOVEAttrPipeline()
    out: Dict[str, Tuple[str, Set[str]]] = {}
    for row in gold:
        obj = pipe.predict(
            current_caption=str(row.get("current_caption") or ""),
            true_image_context=str(row.get("true_image_context") or ""),
            vdt_label=str(row.get("vdt_label") or "OOC"),
            vdt_score=float(row.get("vdt_score") or 0.9),
            sample_id=key_of(row),
            image_id=str(row.get("image_id") or ""),
            domain=str(row.get("domain") or "real_ooc_eval"),
        )
        out[key_of(row)] = (str(obj.get("mismatch_type") or ""), field_set(obj.get("conflict_fields")))
    return out


def prediction_file_predictions(rows: Sequence[Dict[str, Any]], type_key: str, fields_key: str) -> Dict[str, Tuple[str, Set[str]]]:
    out: Dict[str, Tuple[str, Set[str]]] = {}
    for row in rows:
        t = str(row.get(type_key) or "").strip()
        if t:
            out[key_of(row)] = (t, field_set(row.get(fields_key)))
    return out


def attr_head_predictions(feature_rows: Sequence[Dict[str, Any]], model_path: Path) -> Dict[str, Tuple[str, Set[str]]]:
    if not model_path.exists():
        return {}
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    groups = bundle.get("feature_groups") or ("nli", "evidence", "graph")
    X = [feature_vector(row, groups=groups) for row in feature_rows]
    if not X:
        return {}
    type_ids = bundle["type_model"].predict(X)
    types = bundle["type_encoder"].inverse_transform(type_ids)
    field_arr = bundle["field_model"].predict(X)
    out: Dict[str, Tuple[str, Set[str]]] = {}
    fields = list(bundle.get("fields") or FIELDS)
    for row, t, vals in zip(feature_rows, types, field_arr):
        fs = {fields[j] for j, v in enumerate(list(vals)[: len(fields)]) if int(v)}
        out[key_of(row)] = (str(t), fs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate rule/NLI/counterfactual-trained attribution head on manual real OOC gold set.")
    ap.add_argument("--gold", default="examples/real_ooc_attribution_eval_set.jsonl")
    ap.add_argument("--predictions", default="outputs/field_nli_attribution_v2.jsonl")
    ap.add_argument("--attr-head-model", default="outputs/counterfactual/attribution_head_model.pkl")
    ap.add_argument("--output", default="outputs/real_ooc_attribution_eval_metrics.json")
    args = ap.parse_args()

    gold = gold_rows(read_jsonl(Path(args.gold)))
    pred_rows = read_jsonl(Path(args.predictions))
    methods = {
        "rule_sidecar": rule_sidecar_predictions(gold),
        "field_wise_nli": prediction_file_predictions(pred_rows, "v2_mismatch_type", "v2_conflict_fields"),
        "counterfactual_trained_attr_head": attr_head_predictions(pred_rows, Path(args.attr_head_model)),
    }
    results = {name: score(gold, preds) for name, preds in methods.items()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gold": args.gold,
        "gold_records_done": len(gold),
        "predictions": args.predictions,
        "attr_head_model": args.attr_head_model,
        "methods": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "matched", "missing_predictions", "mismatch_type_accuracy", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate"])
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({"method": name, **{k: res.get(k) for k in writer.fieldnames if k != "method"}})
    print(json.dumps({"output": str(out), "csv": str(csv_path), "gold_records_done": len(gold), "methods": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

