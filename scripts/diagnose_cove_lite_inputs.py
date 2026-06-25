from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["annotations", "data", "records", "samples", "items", "examples"]:
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals[: min(10, len(vals))]):
            out = []
            for k, v in obj.items():
                rec = dict(v)
                rec.setdefault("id", k)
                out.append(rec)
            return out
    return []


def preview_json_file(path: Path, limit: int) -> Dict[str, Any]:
    try:
        rows = load_json_records(path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"path": str(path), "error": str(exc)}
    sample = []
    for rec in rows[:limit]:
        sample.append({
            "keys": list(rec.keys())[:20],
            "id": rec.get("id"),
            "image_id": rec.get("image_id") or rec.get("img_id") or rec.get("image"),
            "falsified": rec.get("falsified"),
            "label": rec.get("label"),
            "source_dataset": rec.get("source_dataset"),
        })
    return {"path": str(path), "records": len(rows), "sample": sample}


def preview_pickle_file(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:  # pragma: no cover - diagnostic script
        info["error"] = str(exc)
        return info
    info["object_type"] = type(obj).__name__
    try:
        info["len"] = len(obj)  # type: ignore[arg-type]
    except Exception:
        info["len"] = None
    if isinstance(obj, dict):
        items = list(obj.items())[:2]
        info["sample_keys"] = [str(k) for k, _ in items]
        info["sample_value_types"] = [type(v).__name__ for _, v in items]
        info["sample_value_keys"] = [list(v.keys())[:20] if isinstance(v, dict) else None for _, v in items]
    elif isinstance(obj, list):
        items = obj[:2]
        info["sample_value_types"] = [type(v).__name__ for v in items]
        info["sample_value_keys"] = [list(v.keys())[:20] if isinstance(v, dict) else None for v in items]
    return info


def find_origin_data_json(meta_dir: Path) -> Path | None:
    for path in [meta_dir / "data.json", meta_dir / "origin" / "data.json", meta_dir.parent / "origin" / "data.json"]:
        if path.exists():
            return path
    return None


def preview_origin_data(path: Path | None, limit: int) -> Dict[str, Any] | None:
    if not path:
        return None
    info: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic script
        info["error"] = str(exc)
        return info
    records = obj if isinstance(obj, list) else list(obj.values()) if isinstance(obj, dict) else []
    info["records"] = len(records)
    sample = []
    for rec in records[:limit]:
        if isinstance(rec, dict):
            sample.append({
                "id": rec.get("id"),
                "caption": rec.get("caption"),
                "topic": rec.get("topic"),
                "source": rec.get("source"),
                "image_path": rec.get("image_path"),
                "article_path": rec.get("article_path"),
            })
    info["sample"] = sample
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose local NewsCLIPpings and VisualNews inputs for COVE-lite construction.")
    ap.add_argument("--newsclippings-data-dir", "--newsclippings-dir", dest="newsclippings_data_dir", required=True)
    ap.add_argument("--visualnews-metadata-dir", required=True)
    ap.add_argument("--output", default="outputs/input_check.json")
    ap.add_argument("--preview-files", type=int, default=5)
    ap.add_argument("--preview-records", type=int, default=3)
    args = ap.parse_args()

    news_dir = Path(args.newsclippings_data_dir)
    meta_dir = Path(args.visualnews_metadata_dir)
    json_files = sorted(news_dir.rglob("*.json")) if news_dir.exists() else []
    pickle_files: List[Path] = []
    if meta_dir.exists():
        for pat in ["*.p", "*.pkl", "*.pickle"]:
            pickle_files.extend(meta_dir.rglob(pat))
    pickle_files = sorted(set(pickle_files))
    origin_data = find_origin_data_json(meta_dir)

    result: Dict[str, Any] = {
        "newsclippings_data_dir": str(news_dir),
        "newsclippings_dir_exists": news_dir.exists(),
        "newsclippings_json_count": len(json_files),
        "newsclippings_json_examples": [preview_json_file(p, args.preview_records) for p in json_files[: args.preview_files]],
        "visualnews_metadata_dir": str(meta_dir),
        "visualnews_metadata_dir_exists": meta_dir.exists(),
        "visualnews_pickle_count": len(pickle_files),
        "visualnews_pickle_examples": [
            {"path": str(p), "size_bytes": p.stat().st_size} for p in pickle_files[: args.preview_files]
        ],
        "visualnews_origin_data": preview_origin_data(origin_data, args.preview_records),
        "visualnews_first_pickle_preview": preview_pickle_file(pickle_files[0]) if pickle_files else None,
        "status": "ok",
        "errors": [],
    }

    if not news_dir.exists():
        result["errors"].append("NewsCLIPpings data dir does not exist.")
    if not meta_dir.exists():
        result["errors"].append("VisualNews metadata dir does not exist.")
    if not json_files:
        result["errors"].append("No NewsCLIPpings JSON files found.")
    if not pickle_files and not origin_data:
        result["errors"].append("No VisualNews metadata pickle files or origin/data.json found.")
    if result["errors"]:
        result["status"] = "error"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
