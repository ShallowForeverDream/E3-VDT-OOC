from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(paths: Dict[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    return {split: read_jsonl(path) for split, path in paths.items()}


def values_by_split(rows_by_split: Dict[str, List[Dict[str, Any]]], key: str) -> Dict[str, set]:
    out: Dict[str, set] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            val = str(row.get(key) or "").strip()
            if val:
                out[val].add(split)
    return out


def leakage_report(rows_by_split: Dict[str, List[Dict[str, Any]]], key: str) -> Dict[str, Any]:
    mapping = values_by_split(rows_by_split, key)
    leaked = {k: sorted(v) for k, v in mapping.items() if len(v) > 1}
    return {
        "key": key,
        "leakage": len(leaked),
        "examples": dict(list(leaked.items())[:20]),
    }


def caption_duplicates(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    caption_splits: Dict[str, set] = defaultdict(set)
    caption_counts = Counter()
    for split, rows in rows_by_split.items():
        for row in rows:
            caption = str(row.get("current_caption") or row.get("edited_caption") or "").strip()
            if caption:
                caption_counts[caption] += 1
                caption_splits[caption].add(split)
    duplicates = {k: v for k, v in caption_counts.items() if v > 1}
    cross_split = {k: sorted(caption_splits[k]) for k in duplicates if len(caption_splits[k]) > 1}
    return {
        "duplicate_edited_caption": len(duplicates),
        "cross_split_duplicate_edited_caption": len(cross_split),
        "top_duplicates": dict(caption_counts.most_common(20)),
        "cross_split_examples": dict(list(cross_split.items())[:20]),
    }


def replacement_repetition(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    pairs = Counter()
    for rows in rows_by_split.values():
        for row in rows:
            old = str(row.get("edited_span_text") or row.get("validation", {}).get("original_text") or "").strip()
            new = str(row.get("replacement_text") or row.get("validation", {}).get("replacement_text") or "").strip()
            if old or new:
                pairs[f"{old} -> {new}"] += 1
    repeated = {k: v for k, v in pairs.items() if v > 1}
    return {
        "unique_replacement_pairs": len(pairs),
        "repeated_replacement_pairs": len(repeated),
        "top_replacement_pairs": dict(pairs.most_common(20)),
    }


def class_distribution(rows_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        all_rows.extend(rows)
        out[split] = {
            "rows": len(rows),
            "edit_type": dict(Counter(str(r.get("edit_type") or "") for r in rows)),
            "gold_mismatch_type": dict(Counter(str(r.get("gold_mismatch_type") or "") for r in rows)),
        }
    out["all"] = {
        "rows": len(all_rows),
        "edit_type": dict(Counter(str(r.get("edit_type") or "") for r in all_rows)),
        "gold_mismatch_type": dict(Counter(str(r.get("gold_mismatch_type") or "") for r in all_rows)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check controlled counterfactual train/val/test leakage.")
    ap.add_argument("--train", default="outputs/counterfactual/controlled_counterfactual_train.jsonl")
    ap.add_argument("--val", default="outputs/counterfactual/controlled_counterfactual_val.jsonl")
    ap.add_argument("--test", default="outputs/counterfactual/controlled_counterfactual_test.jsonl")
    ap.add_argument("--output", default="outputs/counterfactual/leakage_check.json")
    ap.add_argument("--fail-on-leak", action="store_true")
    args = ap.parse_args()

    rows_by_split = split_rows({"train": Path(args.train), "val": Path(args.val), "test": Path(args.test)})
    source = leakage_report(rows_by_split, "source_sample_id")
    image = leakage_report(rows_by_split, "image_id")
    text = leakage_report(rows_by_split, "text_id")
    dup = caption_duplicates(rows_by_split)
    repl = replacement_repetition(rows_by_split)
    classes = class_distribution(rows_by_split)
    result = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "source_sample_id_leakage": source["leakage"],
        "image_id_leakage": image["leakage"],
        "text_id_leakage": text["leakage"],
        "duplicate_edited_caption": dup["duplicate_edited_caption"],
        "cross_split_duplicate_edited_caption": dup["cross_split_duplicate_edited_caption"],
        "class_counts": classes["all"]["gold_mismatch_type"],
        "edit_type_counts": classes["all"]["edit_type"],
        "details": {
            "source_sample_id": source,
            "image_id": image,
            "text_id": text,
            "captions": dup,
            "replacement_pairs": repl,
            "class_distribution": classes,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_leak and (source["leakage"] or image["leakage"] or text["leakage"] or dup["cross_split_duplicate_edited_caption"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

