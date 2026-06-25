from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


FIELDS = ["entity", "location", "time", "event_type", "relation"]
TYPE_TO_FIELDS = {
    "none": [],
    "benign illustrative image": [],
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "different-event mismatch": ["entity", "location", "event_type"],
}
META_COLUMNS = {
    "sample_id",
    "source_sample_id",
    "image_id",
    "domain",
    "split",
    "current_caption",
    "image_path",
    "gold_mismatch_type",
    "entity_prompt",
    "location_prompt",
    "time_prompt",
    "event_type_prompt",
    "relation_prompt",
}


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def normalize_type(t: str) -> str:
    t = str(t or "").strip()
    return "benign illustrative image" if t == "none" else t


def feature_names(rows: Sequence[Dict[str, Any]], groups: Sequence[str] = ("clip", "field", "vdt")) -> List[str]:
    if not rows:
        return []
    names = []
    for k in rows[0].keys():
        if k in META_COLUMNS or k.startswith("gold_field_"):
            continue
        if "clip" not in groups and k.startswith("clip_"):
            continue
        if "field" not in groups and (k.endswith("_count") or k.endswith("_present") or k in {"caption_chars", "caption_tokens", "image_loaded"}):
            continue
        if "vdt" not in groups and k == "vdt_score":
            continue
        # Keep numeric-looking columns only.
        if any(ch.isalpha() for ch in str(rows[0].get(k, ""))) and not str(rows[0].get(k, "")).replace(".", "", 1).replace("-", "", 1).isdigit():
            continue
        names.append(k)
    return names


def matrix(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.array([[as_float(r.get(k)) for k in names] for r in rows], dtype=float)


def labels(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
    y_type: List[str] = []
    y_fields = np.zeros((len(rows), len(FIELDS)), dtype=int)
    for i, row in enumerate(rows):
        t = normalize_type(str(row.get("gold_mismatch_type") or ""))
        if not t:
            t = "uncertain / evidence insufficient"
        y_type.append(t)
        inferred = set(TYPE_TO_FIELDS.get(t, []))
        for j, field in enumerate(FIELDS):
            y_fields[i, j] = int(as_float(row.get(f"gold_field_{field}")) > 0 or field in inferred)
    return y_type, y_fields


def type_to_field_set(t: str) -> Set[str]:
    return set(TYPE_TO_FIELDS.get(t, []))


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


class SafeFieldLogReg:
    def __init__(self, seed: int = 2026) -> None:
        self.seed = seed
        self.models: List[Any] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SafeFieldLogReg":
        self.models = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if len(set(y.tolist())) < 2:
                model = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
            else:
                model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed + j)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cols = [m.predict(X) for m in self.models]
        return np.vstack(cols).T if cols else np.zeros((len(X), 0), dtype=int)


def train_head(train_rows: Sequence[Dict[str, Any]], test_rows: Sequence[Dict[str, Any]], groups: Sequence[str], kind: str, seed: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
    names = feature_names(train_rows, groups=groups)
    X_train = matrix(train_rows, names)
    X_test = matrix(test_rows, names)
    y_type_train, y_fields_train = labels(train_rows)
    y_type_test, y_fields_test = labels(test_rows)
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y_type_train)
    if kind == "logreg":
        type_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
        field_model = make_pipeline(StandardScaler(), SafeFieldLogReg(seed=seed + 1))
    else:
        type_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), random_state=seed, max_iter=600, early_stopping=False))
        field_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), random_state=seed + 1, max_iter=600, early_stopping=False))
    type_model.fit(X_train, y_enc)
    field_model.fit(X_train, y_fields_train)
    pred_types = list(enc.inverse_transform(type_model.predict(X_test)))
    field_arr = np.array(field_model.predict(X_test), dtype=int)
    if field_arr.ndim == 1:
        field_arr = field_arr.reshape(-1, 1)
    pred_fields = [{FIELDS[j] for j, v in enumerate(row[: len(FIELDS)]) if v} for row in field_arr]
    bundle = {
        "type_model": type_model,
        "type_encoder": enc,
        "field_model": field_model,
        "fields": FIELDS,
        "feature_names": names,
        "feature_groups": list(groups),
        "head_kind": kind,
        "uses_true_context_at_inference": False,
    }
    return score_predictions(y_type_test, y_fields_test, pred_types, pred_fields), bundle


def prompt_rule(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Set[str]]]:
    pred_types: List[str] = []
    pred_fields: List[Set[str]] = []
    for r in rows:
        sims = {
            f: as_float(r.get(f"clip_prompt_sim_{f}"))
            for f in FIELDS
            if as_float(r.get(f"{f}_present")) > 0
        }
        if not sims or as_float(r.get("image_loaded")) <= 0:
            pred_types.append("uncertain / evidence insufficient")
            pred_fields.append(set())
            continue
        # CLIP similarities are not calibrated; use relative minimum as a weak rule.
        field = min(sims, key=sims.get)
        spread = max(sims.values()) - min(sims.values())
        if spread < 0.015:
            pred_types.append("benign illustrative image")
            pred_fields.append(set())
        else:
            t = {
                "entity": "entity mismatch",
                "location": "location mismatch",
                "time": "temporal mismatch",
                "event_type": "event-type mismatch",
                "relation": "relation mismatch",
            }[field]
            pred_types.append(t)
            pred_fields.append({field})
    return pred_types, pred_fields


def main() -> None:
    ap = argparse.ArgumentParser(description="Train no-true-context image+caption attribution head.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model-out", default="outputs/no_true_context_attr/no_true_context_attr_head.pkl")
    ap.add_argument("--metrics-out", default="outputs/no_true_context_attr/no_true_context_attr_metrics.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_rows = read_csv(Path(args.train))
    val_rows = read_csv(Path(args.val))
    test_rows = read_csv(Path(args.test))
    train_all = train_rows + val_rows
    y_type_train, _ = labels(train_all)
    y_type_test, y_fields_test = labels(test_rows)

    majority = Counter(y_type_train).most_common(1)[0][0] if y_type_train else "benign illustrative image"
    majority_types = [majority for _ in test_rows]
    majority_fields = [type_to_field_set(majority) for _ in test_rows]
    prompt_types, prompt_fields = prompt_rule(test_rows)
    results: Dict[str, Any] = {
        "majority": score_predictions(y_type_test, y_fields_test, majority_types, majority_fields),
        "field_prompt_grounding_rule": score_predictions(y_type_test, y_fields_test, prompt_types, prompt_fields),
    }
    variants = {
        "logistic_regression_no_true_context": (("clip", "field", "vdt"), "logreg"),
        "mlp_no_clip_prompt": (("field", "vdt"), "mlp"),
        "mlp_no_field_presence": (("clip", "vdt"), "mlp"),
        "attr_head_image_caption_mlp": (("clip", "field", "vdt"), "mlp"),
    }
    bundle: Dict[str, Any] = {}
    for name, (groups, kind) in variants.items():
        res, b = train_head(train_all, test_rows, groups=groups, kind=kind, seed=args.seed + len(name))
        results[name] = res
        if name == "attr_head_image_caption_mlp":
            bundle = b

    metrics = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "uses_true_context_at_inference": False,
        "type_classes": sorted(set(y_type_train)),
        "field_labels": FIELDS,
        "results": results,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(bundle, f)
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "uses_true_context_at_inference", "mismatch_type_accuracy", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate"])
        writer.writeheader()
        for name, res in results.items():
            writer.writerow({"method": name, "uses_true_context_at_inference": False, **{k: res.get(k) for k in writer.fieldnames if k not in {"method", "uses_true_context_at_inference"}}})
    print(json.dumps({"model": str(model_out), "metrics": str(out), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

