from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

ALL_FIELDS = ["entity", "location", "time", "event_type", "relation", "context_omission", "evidence_insufficient"]
TYPE_TO_FIELDS = {
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "context omission": ["context_omission"],
    "uncertain / evidence insufficient": ["evidence_insufficient"],
    "benign illustrative image": [],
}
MISMATCH_TYPES = list(TYPE_TO_FIELDS.keys())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_type(x: Any) -> str:
    s = str(x or "").strip().lower().replace("_", " ")
    aliases = {
        "location": "location mismatch",
        "temporal": "temporal mismatch",
        "time mismatch": "temporal mismatch",
        "event mismatch": "event-type mismatch",
        "event type mismatch": "event-type mismatch",
        "insufficient": "uncertain / evidence insufficient",
        "uncertain": "uncertain / evidence insufficient",
        "none": "benign illustrative image",
    }
    if s in aliases:
        return aliases[s]
    for t in MISMATCH_TYPES:
        if s == t.lower():
            return t
    return s if s else "uncertain / evidence insufficient"


def norm_field(x: Any) -> str:
    s = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "event": "event_type",
        "event_type": "event_type",
        "temporal": "time",
        "person": "entity",
        "subject": "entity",
        "place": "location",
        "uncertain": "evidence_insufficient",
        "insufficient": "evidence_insufficient",
        "omission": "context_omission",
    }
    return aliases.get(s, s)


def field_set(xs: Any) -> Set[str]:
    if xs is None:
        return set()
    if isinstance(xs, str):
        parts = [p.strip() for p in xs.replace(";", ",").split(",") if p.strip()]
    elif isinstance(xs, list):
        parts = xs
    else:
        parts = [xs]
    return {norm_field(x) for x in parts if norm_field(x) in ALL_FIELDS}


def fields_from_type(t: str) -> Set[str]:
    return set(TYPE_TO_FIELDS.get(norm_type(t), []))


def score(gold_rows: List[Dict[str, Any]], pred_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total = type_ok = exact = 0
    tp = fp = fn = 0
    macro = []
    per = {f: {"tp": 0, "fp": 0, "fn": 0} for f in ALL_FIELDS}

    for g in gold_rows:
        if not g.get("gold_mismatch_type"):
            continue
        sid = str(g.get("sample_id", ""))
        p = pred_by_id.get(sid, {})
        gt_type = norm_type(g.get("gold_mismatch_type"))
        pr_type = norm_type(p.get("mismatch_type"))
        gt_fields = field_set(g.get("gold_conflict_fields")) or fields_from_type(gt_type)
        pr_fields = field_set(p.get("conflict_fields")) or fields_from_type(pr_type)
        total += 1
        type_ok += int(gt_type == pr_type)
        exact += int(gt_fields == pr_fields)
        for f in ALL_FIELDS:
            ing, inp = f in gt_fields, f in pr_fields
            if ing and inp:
                tp += 1; per[f]["tp"] += 1
            elif inp and not ing:
                fp += 1; per[f]["fp"] += 1
            elif ing and not inp:
                fn += 1; per[f]["fn"] += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    for f, c in per.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        macro.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "n": total,
        "mismatch_type_accuracy": type_ok / total if total else 0.0,
        "conflict_field_micro_precision": prec,
        "conflict_field_micro_recall": rec,
        "conflict_field_micro_f1": f1,
        "conflict_field_macro_f1": sum(macro) / len(macro) if macro else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
    }


def majority_preds(gold_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    counts = Counter(norm_type(r.get("gold_mismatch_type")) for r in gold_rows if r.get("gold_mismatch_type"))
    maj = counts.most_common(1)[0][0] if counts else "uncertain / evidence insufficient"
    return {str(r.get("sample_id")): {"mismatch_type": maj, "conflict_fields": list(fields_from_type(maj))} for r in gold_rows}


def sampled_preds(gold_rows: List[Dict[str, Any]], seed: int) -> Dict[str, Dict[str, Any]]:
    rng = random.Random(seed)
    out = {}
    for r in gold_rows:
        t = rng.choice(MISMATCH_TYPES)
        out[str(r.get("sample_id"))] = {"mismatch_type": t, "conflict_fields": list(fields_from_type(t))}
    return out


def text_only_preds(gold_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    t = "uncertain / evidence insufficient"
    return {str(r.get("sample_id")): {"mismatch_type": t, "conflict_fields": ["evidence_insufficient"]} for r in gold_rows}


def weak_preds(weak_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in weak_rows:
        sid = str(r.get("sample_id", ""))
        if sid:
            out[sid] = {"mismatch_type": r.get("weak_mismatch_type"), "conflict_fields": r.get("weak_conflict_fields", [])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--weak-labels", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    gold = read_jsonl(Path(args.gold))
    weak = read_jsonl(Path(args.weak_labels)) if args.weak_labels else []
    methods = {
        "majority": majority_preds(gold),
        "sampled": sampled_preds(gold, args.seed),
        "text_only_uncertain": text_only_preds(gold),
    }
    if weak:
        methods["cove_lite_event_rule"] = weak_preds(weak)

    results = {name: score(gold, preds) for name, preds in methods.items()}
    out = {"gold": args.gold, "weak_labels": args.weak_labels, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    keys = ["n", "mismatch_type_accuracy", "conflict_field_micro_precision", "conflict_field_micro_recall", "conflict_field_micro_f1", "conflict_field_macro_f1", "exact_match_rate"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("method," + ",".join(keys) + "\n")
        for name, m in results.items():
            f.write(name + "," + ",".join(str(round(float(m.get(k, 0.0)), 6)) for k in keys) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
