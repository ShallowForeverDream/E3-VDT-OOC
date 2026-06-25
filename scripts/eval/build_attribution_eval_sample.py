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


def round_robin_sample(buckets: Dict[str, List[Dict[str, Any]]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    for rows in buckets.values():
        rng.shuffle(rows)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    selected: List[Dict[str, Any]] = []
    while len(selected) < n and any(buckets.values()):
        for key in list(keys):
            if len(selected) >= n:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
    return selected


def by_type(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[str(r.get("weak_mismatch_type", "unknown"))].append(r)
    return buckets


def enough_balance(records: List[Dict[str, Any]], n: int) -> bool:
    if not records:
        return False
    ooc = sum(1 for r in records if r.get("label") == 1)
    non = sum(1 for r in records if r.get("label") == 0)
    # Prefer split only if it can provide a useful attribution set.
    return len(records) >= n and ooc >= max(8, n // 4) and non >= max(8, n // 4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample candidate records for manual attribution annotation.")
    ap.add_argument("--context-pairs", required=True)
    ap.add_argument("--weak-labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefer-split", default="test")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.weak_labels))
    rng = random.Random(args.seed)

    preferred = [r for r in rows if str(r.get("split", "")).lower() == args.prefer_split.lower()]
    if enough_balance(preferred, args.n):
        pool = preferred
        pool_source = f"preferred_split={args.prefer_split}"
    else:
        pool = rows
        pool_source = "all_splits_fallback"

    ooc = [r for r in pool if r.get("label") == 1]
    non = [r for r in pool if r.get("label") == 0]
    other = [r for r in pool if r.get("label") not in {0, 1}]

    # Aim for half OOC / half Non-OOC if the pool allows it.
    target_ooc = args.n // 2
    n_ooc = min(len(ooc), target_ooc)
    n_non = min(len(non), args.n - n_ooc)
    n_other = max(0, args.n - n_ooc - n_non)

    selected: List[Dict[str, Any]] = []
    selected.extend(round_robin_sample(by_type(ooc), n_ooc, rng))
    selected.extend(round_robin_sample(by_type(non), n_non, rng))
    if n_other:
        selected.extend(round_robin_sample(by_type(other), n_other, rng))
    rng.shuffle(selected)

    out_rows: List[Dict[str, Any]] = []
    for r in selected[: args.n]:
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
    print(json.dumps({
        "output": args.output,
        "records": len(out_rows),
        "pool_source": pool_source,
        "pool_size": len(pool),
        "pool_ooc": len(ooc),
        "pool_non_ooc": len(non),
        "selected_ooc": sum(1 for r in out_rows if r.get("label") == 1),
        "selected_non_ooc": sum(1 for r in out_rows if r.get("label") == 0),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
