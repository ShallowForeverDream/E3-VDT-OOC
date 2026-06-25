from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
from typing import Any


def safe_len(x: Any):
    try:
        return len(x)
    except Exception:
        return None


def summarize_pickle(path: Path) -> dict:
    info = {"file": str(path), "bytes": path.stat().st_size, "type": None, "len": None, "sample": None, "error": None}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        info["type"] = type(obj).__name__
        info["len"] = safe_len(obj)
        sample = None
        if hasattr(obj, "head"):
            # pandas DataFrame-like
            sample = obj.head(2).to_dict(orient="records")
        elif isinstance(obj, dict):
            keys = list(obj.keys())[:5]
            sample = {str(k): type(obj[k]).__name__ for k in keys}
        elif isinstance(obj, (list, tuple)):
            sample = [type(x).__name__ for x in list(obj)[:5]]
        info["sample"] = sample
    except Exception as e:
        info["error"] = repr(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-dir", required=True)
    ap.add_argument("--visual-dir", required=True)
    ap.add_argument("--out", default="outputs/input_check.json")
    args = ap.parse_args()
    news_dir = Path(args.news_dir)
    visual_dir = Path(args.visual_dir)
    json_files = sorted(news_dir.rglob("*.json")) if news_dir.exists() else []
    jsonl_files = sorted(news_dir.rglob("*.jsonl")) if news_dir.exists() else []
    meta_files = []
    if visual_dir.exists():
        for pat in ["processed_*.p", "*.pkl", "*.pickle", "*.p"]:
            meta_files.extend(sorted(visual_dir.rglob(pat)))
    seen = set(); uniq_meta=[]
    for p in meta_files:
        if p not in seen:
            seen.add(p); uniq_meta.append(p)
    out = {
        "news_dir": str(news_dir),
        "visual_dir": str(visual_dir),
        "news_json_count": len(json_files),
        "news_jsonl_count": len(jsonl_files),
        "news_json_examples": [str(p) for p in json_files[:10]],
        "metadata_file_count": len(uniq_meta),
        "metadata_files": [summarize_pickle(p) for p in uniq_meta[:20]],
    }
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
