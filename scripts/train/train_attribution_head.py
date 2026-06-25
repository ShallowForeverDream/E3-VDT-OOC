from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


FIELDS = ["entity", "location", "time", "event_type", "relation", "context_omission", "evidence_insufficient"]
TYPE_TO_FIELDS = {
    "benign illustrative image": [],
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "different-event mismatch": ["entity", "location", "event_type"],
    "context omission": ["context_omission"],
    "uncertain / evidence insufficient": ["evidence_insufficient"],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def field_set(xs: Any) -> Set[str]:
    if isinstance(xs, str):
        parts = [x.strip() for x in xs.replace(";", ",").split(",")]
    else:
        parts = list(xs or [])
    return {str(x).strip() for x in parts if str(x).strip()}


def get_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def feature_vector(row: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    field_nli = row.get("field_nli") or {}
    for field in ["entity", "location", "time", "event_type", "relation"]:
        item = field_nli.get(field, {}) if isinstance(field_nli, dict) else {}
        scores = item.get("scores", {}) if isinstance(item, dict) else {}
        out.extend([
            get_num(scores.get("entailment")),
            get_num(scores.get("neutral")),
            get_num(scores.get("contradiction")),
        ])
    evidence = row.get("evidence_relevance") or {}
    overlaps = evidence.get("field_overlaps", {}) if isinstance(evidence, dict) else {}
    out.extend([
        get_num(evidence.get("evidence_relevance")),
        get_num(evidence.get("text_similarity")),
        get_num(overlaps.get("entity")),
        get_num(overlaps.get("location")),
        get_num(overlaps.get("time")),
        get_num(overlaps.get("event_type")),
        get_num(overlaps.get("relation")),
        get_num(evidence.get("filled_true_fields")),
        get_num(evidence.get("filled_current_fields")),
        get_num(evidence.get("context_length")),
    ])
    graph = row.get("graph_alignment") or {}
    out.extend([
        get_num(graph.get("graph_alignment_score")),
        get_num(graph.get("num_current_edges")),
        get_num(graph.get("num_true_edges")),
        1.0 if "relation" in field_set(graph.get("graph_conflicts", [])) else 0.0,
    ])
    return out


def labels(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
    y_type: List[str] = []
    y_fields = np.zeros((len(rows), len(FIELDS)), dtype=int)
    for i, row in enumerate(rows):
        t = str(row.get("gold_mismatch_type") or row.get("v2_mismatch_type") or "").strip()
        if not t:
            t = "uncertain / evidence insufficient"
        y_type.append(t)
        fields = field_set(row.get("gold_conflict_fields"))
        if not fields:
            fields = set(TYPE_TO_FIELDS.get(t, []))
        for j, f in enumerate(FIELDS):
            y_fields[i, j] = int(f in fields)
    return y_type, y_fields


def predictions_from_rows(rows: Sequence[Dict[str, Any]], type_key: str, fields_key: str) -> Tuple[List[str], List[Set[str]]]:
    types = [str(r.get(type_key, "")).strip() for r in rows]
    fields = [field_set(r.get(fields_key)) for r in rows]
    return types, fields


def score_predictions(gold_types: Sequence[str], gold_fields_arr: np.ndarray, pred_types: Sequence[str], pred_fields: Sequence[Set[str]]) -> Dict[str, float]:
    gold_field_sets = [{FIELDS[j] for j, v in enumerate(row) if v} for row in gold_fields_arr]
    pred_arr = np.zeros_like(gold_fields_arr)
    for i, fs in enumerate(pred_fields):
        for j, f in enumerate(FIELDS):
            pred_arr[i, j] = int(f in fs)
    return {
        "n": len(gold_types),
        "mismatch_type_accuracy": float(accuracy_score(gold_types, pred_types)) if gold_types else 0.0,
        "conflict_field_micro_f1": float(f1_score(gold_fields_arr, pred_arr, average="micro", zero_division=0)) if len(gold_types) else 0.0,
        "conflict_field_macro_f1": float(f1_score(gold_fields_arr, pred_arr, average="macro", zero_division=0)) if len(gold_types) else 0.0,
        "exact_match_rate": float(sum(1 for g, p in zip(gold_field_sets, pred_fields) if g == p) / len(gold_field_sets)) if gold_field_sets else 0.0,
    }


def type_to_field_set(t: str) -> Set[str]:
    return set(TYPE_TO_FIELDS.get(t, []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a lightweight attribution head on controlled counterfactual features.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model-out", default="outputs/attribution_head_model.pkl")
    ap.add_argument("--metrics-out", default="outputs/attribution_head_metrics.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_rows = read_jsonl(Path(args.train))
    val_rows = read_jsonl(Path(args.val))
    test_rows = read_jsonl(Path(args.test))
    train_all = train_rows + val_rows
    X_train = np.array([feature_vector(r) for r in train_all], dtype=float)
    X_test = np.array([feature_vector(r) for r in test_rows], dtype=float)
    y_type_train, y_fields_train = labels(train_all)
    y_type_test, y_fields_test = labels(test_rows)
    type_encoder = LabelEncoder()
    y_type_train_enc = type_encoder.fit_transform(y_type_train)

    type_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), random_state=args.seed, max_iter=600, early_stopping=False),
    )
    field_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), random_state=args.seed + 1, max_iter=600, early_stopping=False),
    )
    type_model.fit(X_train, y_type_train_enc)
    field_model.fit(X_train, y_fields_train)

    type_pred = list(type_encoder.inverse_transform(type_model.predict(X_test)))
    field_pred_arr = np.array(field_model.predict(X_test), dtype=int)
    field_pred = [{FIELDS[j] for j, v in enumerate(row) if v} for row in field_pred_arr]

    majority = Counter(y_type_train).most_common(1)[0][0] if y_type_train else "benign illustrative image"
    majority_types = [majority for _ in test_rows]
    majority_fields = [type_to_field_set(majority) for _ in test_rows]
    nli_types, nli_fields = predictions_from_rows(test_rows, "v2_mismatch_type", "v2_conflict_fields")
    weak_types, weak_fields = predictions_from_rows(test_rows, "weak_mismatch_type", "weak_conflict_fields")

    results = {
        "majority": score_predictions(y_type_test, y_fields_test, majority_types, majority_fields),
        "field_wise_nli": score_predictions(y_type_test, y_fields_test, nli_types, nli_fields),
        "weak_rule_sidecar": score_predictions(y_type_test, y_fields_test, weak_types, weak_fields) if any(weak_types) else None,
        "attr_head_mlp": score_predictions(y_type_test, y_fields_test, type_pred, field_pred),
    }
    results = {k: v for k, v in results.items() if v is not None}
    metrics = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "feature_dim": int(X_train.shape[1]) if X_train.size else 0,
        "type_classes": list(type_encoder.classes_),
        "field_labels": FIELDS,
        "results": results,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump({
            "type_model": type_model,
            "type_encoder": type_encoder,
            "field_model": field_model,
            "fields": FIELDS,
            "type_to_fields": TYPE_TO_FIELDS,
        }, f)
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "mismatch_type_accuracy", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate"])
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({"method": name, **{k: res.get(k) for k in writer.fieldnames if k != "method"}})
    print(json.dumps({"model": str(model_out), "metrics": str(out), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
