from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample candidate records for manual attribution annotation.")
    ap.add_argument("--context-pairs", required=True)
    ap.add_argument("--weak-labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefer-split", default="test")
    args = ap.parse_args()

    weak_rows = read_jsonl(Path(args.weak_labels))
    rng = random.Random(args.seed)

    # Prefer test split, then all rows.
    preferred = [r for r in weak_rows if str(r.get("split", "")).lower() == args.prefer_split.lower()]
    pool = preferred if len(preferred) >= min(args.n, 20) else weak_rows

    # Stratify by label and weak mismatch type to avoid all samples being one type.
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        key = f"label={r.get('label')}|type={r.get('weak_mismatch_type','unknown')}"
        buckets[key].append(r)

    for rows in buckets.values():
        rng.shuffle(rows)

    selected: List[Dict[str, Any]] = []
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    while len(selected) < args.n and any(buckets.values()):
        for key in list(bucket_keys):
            if buckets[key] and len(selected) < args.n:
                selected.append(buckets[key].pop())

    # Convert to annotation template.
    out_rows: List[Dict[str, Any]] = []
    for r in selected:
        out_rows.append({
            "sample_id": r.get("sample_id", ""),
            "image_id": r.get("image_id", ""),
            "text_id": r.get("text_id", ""),
            "split": r.get("split", ""),
            "generator": r.get("generator", ""),
            "domain": r.get("domain", ""),
            "label": r.get("label", None),
            "current_caption": r.get("current_caption", ""),
            "true_image_context": r.get("true_image_context", ""),
            "weak_mismatch_type": r.get("weak_mismatch_type", ""),
            "weak_conflict_fields": r.get("weak_conflict_fields", []),
            "event_scores": r.get("event_scores", {}),
            "gold_mismatch_type": "",
            "gold_conflict_fields": [],
            "annotator": "",
            "rationale": "",
            "annotation_status": "todo"
        })

    write_jsonl(Path(args.output), out_rows)
    print(json.dumps({"output": args.output, "records": len(out_rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
