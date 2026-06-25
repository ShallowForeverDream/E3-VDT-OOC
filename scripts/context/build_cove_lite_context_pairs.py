from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEXT_KEYS = [
    "caption", "caption_text", "text", "sentence", "claim", "title", "headline",
    "description", "article_title", "article", "body", "context", "abstract",
]
ID_KEYS = ["id", "text_id", "caption_id", "ann_id", "annotation_id"]
IMAGE_KEYS = ["image_id", "img_id", "image", "imageId"]
LABEL_KEYS = ["falsified", "label", "is_falsified", "fake", "target"]
SOURCE_KEYS = ["source", "source_name", "news_source", "domain", "publisher", "site", "source_domain"]


def norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(jpg|jpeg|png|webp)$", "", s, flags=re.I)
    return s


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["annotations", "data", "records", "samples", "items", "examples"]:
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        # Some files are dict[id] -> record.
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals[:10]):
            out = []
            for k, v in obj.items():
                rec = dict(v)
                rec.setdefault("id", k)
                out.append(rec)
            return out
    raise ValueError(f"Unsupported JSON structure: {path}")


def iter_newsclippings_json(data_dir: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for path in sorted(data_dir.glob("*.json")):
        try:
            records = load_json_records(path)
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}", file=sys.stderr)
            continue
        for rec in records:
            yield path, rec


def get_first(rec: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in rec and rec[key] not in [None, ""]:
            return rec[key]
    return None


def derive_split_generator(path: Path) -> Tuple[str, str]:
    name = path.stem.lower()
    split = "unknown"
    for s in ["train", "val", "test", "valid", "validation"]:
        if re.search(rf"(^|[_\-.]){s}($|[_\-.])", name):
            split = "val" if s in {"valid", "validation"} else s
            break
    generator = name
    for s in ["train", "val", "test", "valid", "validation"]:
        generator = re.sub(rf"(^|[_\-.]){s}($|[_\-.])", "_", generator)
    generator = generator.strip("_.-") or name
    return split, generator


def normalize_label(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "fake", "falsified", "ooc", "out-of-context", "out_of_context"}:
        return 1
    if s in {"0", "false", "real", "non-ooc", "non_ooc", "truthful", "match", "matched"}:
        return 0
    return None


def flatten_texts(obj: Any, depth: int = 0) -> List[str]:
    if depth > 3 or obj is None:
        return []
    if isinstance(obj, str):
        s = re.sub(r"\s+", " ", obj).strip()
        return [s] if len(s) >= 3 else []
    if isinstance(obj, (int, float, bool)):
        return []
    out: List[str] = []
    if isinstance(obj, dict):
        # Prefer known text keys first.
        for key in TEXT_KEYS:
            if key in obj:
                out.extend(flatten_texts(obj[key], depth + 1))
        # Then search selected nested dictionaries.
        for key, val in obj.items():
            if key in TEXT_KEYS:
                continue
            if key.lower() in {"metadata", "article", "caption", "context", "news", "source"}:
                out.extend(flatten_texts(val, depth + 1))
    elif isinstance(obj, list):
        for val in obj[:5]:
            out.extend(flatten_texts(val, depth + 1))
    return out


def best_context(obj: Any, max_chars: int = 420) -> str:
    texts = []
    for t in flatten_texts(obj):
        if t and t not in texts:
            texts.append(t)
    if not texts:
        return ""
    # Prefer concise caption/title-like texts over giant articles.
    texts.sort(key=lambda s: (0 if 20 <= len(s) <= 260 else 1, len(s)))
    joined = " ".join(texts[:3])
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined[:max_chars]


def find_source(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in SOURCE_KEYS:
            if key in obj and obj[key]:
                return str(obj[key]).strip().lower().replace(" ", "_")
        for val in obj.values():
            if isinstance(val, dict):
                src = find_source(val)
                if src:
                    return src
    return "unknown"


def load_pickle(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def iter_metadata_files(metadata_dir: Path) -> List[Path]:
    patterns = ["*.p", "*.pkl", "*.pickle", "*.p.gz", "*.pkl.gz"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(metadata_dir.rglob(pat))
    return sorted(set(files))


def build_metadata_index(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    files = iter_metadata_files(metadata_dir)
    if not files:
        raise FileNotFoundError(f"No pickle metadata files found under {metadata_dir}")
    for path in files:
        try:
            obj = load_pickle(path)
        except Exception as exc:
            print(f"[WARN] cannot load metadata {path}: {exc}", file=sys.stderr)
            continue
        if isinstance(obj, dict):
            items = obj.items()
        elif isinstance(obj, list):
            items = enumerate(obj)
        else:
            continue
        for key, val in items:
            # Direct key often is image/article id.
            keys = {norm_id(key)}
            if isinstance(val, dict):
                for k in ["id", "image_id", "img_id", "uid", "article_id"]:
                    if k in val:
                        keys.add(norm_id(val.get(k)))
            context = best_context(val)
            if not context:
                continue
            src = find_source(val)
            entry = {"context": context, "source": src, "raw_key": str(key), "metadata_file": str(path)}
            for k in keys:
                if k and k not in index:
                    index[k] = entry
    return index


def rec_text(rec: Dict[str, Any], meta: Dict[str, Dict[str, Any]], text_id: str) -> str:
    direct = best_context(rec)
    # If direct text is only id-like or absent, use metadata by text_id.
    if direct and len(direct.split()) >= 3:
        return direct
    if text_id in meta:
        return meta[text_id]["context"]
    return direct


def maybe_sample(records: List[Dict[str, Any]], max_records: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    rng = random.Random(seed)
    return rng.sample(records, max_records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build COVE-lite current-caption vs true-image-context pairs.")
    ap.add_argument("--newsclippings-data-dir", required=True, help="Directory containing NewsCLIPpings JSON files.")
    ap.add_argument("--visualnews-metadata-dir", required=True, help="Directory containing VisualNews processed_*.p / pickle metadata.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="", help="Optional split filter: train/val/test.")
    ap.add_argument("--generator", default="", help="Optional generator filename substring filter, e.g. merged_balanced.")
    ap.add_argument("--max-records", type=int, default=0, help="Optional max records for quick smoke tests.")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    data_dir = Path(args.newsclippings_data_dir)
    meta_dir = Path(args.visualnews_metadata_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading VisualNews metadata from {meta_dir}", file=sys.stderr)
    meta = build_metadata_index(meta_dir)
    print(f"[INFO] metadata index size={len(meta)}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "missing_ids": 0, "missing_text": 0, "missing_true_context": 0}

    for path, rec in iter_newsclippings_json(data_dir):
        split, generator = derive_split_generator(path)
        if args.split and split != args.split:
            continue
        if args.generator and args.generator.lower() not in generator.lower():
            continue
        stats["total"] += 1
        sample_id = norm_id(get_first(rec, ID_KEYS))
        image_id = norm_id(get_first(rec, IMAGE_KEYS))
        text_id = sample_id
        label = normalize_label(get_first(rec, LABEL_KEYS))
        if label is None:
            label = normalize_label(rec.get("falsified"))
        if not sample_id or not image_id:
            stats["missing_ids"] += 1
            continue
        current_caption = rec_text(rec, meta, text_id)
        true_entry = meta.get(image_id, {})
        true_context = true_entry.get("context", "")
        if not current_caption:
            stats["missing_text"] += 1
        if not true_context:
            stats["missing_true_context"] += 1
        if not current_caption or not true_context:
            continue
        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "text_id": text_id,
            "split": split,
            "generator": generator,
            "domain": true_entry.get("source", "unknown"),
            "label": int(label) if label is not None else None,
            "current_caption": current_caption,
            "true_image_context": true_context,
            "source": "visualnews_metadata",
            "newsclippings_file": str(path),
        })

    rows = maybe_sample(rows, args.max_records, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["kept"] = len(rows)
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "stats": stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
