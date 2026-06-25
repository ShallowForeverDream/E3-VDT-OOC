from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEXT_KEY_HINTS = [
    "caption", "title", "headline", "description", "text", "sentence", "claim",
    "article", "body", "content", "context", "abstract", "summary",
]
ID_KEY_HINTS = ["id", "image", "img", "photo", "uid"]
LABEL_KEYS = ["falsified", "label", "is_falsified", "fake", "target"]
IMAGE_KEYS = ["image_id", "img_id", "image", "imageId", "img"]
TEXT_ID_KEYS = ["id", "text_id", "caption_id", "ann_id", "annotation_id", "article_id"]


def norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace("\\", "/")
    if not s:
        return ""
    s = s.split("?")[0]
    s = s.split("/")[-1]
    s = re.sub(r"\.(jpg|jpeg|png|webp|gif|bmp)$", "", s, flags=re.I)
    return s.strip()


def is_text_like(s: str) -> bool:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) < 8:
        return False
    if len(s) > 2000:
        return False
    low = s.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return False
    if re.fullmatch(r"[\w\-./\\]+", s) and not re.search(r"[a-zA-Z]{3,}\s+[a-zA-Z]{3,}", s):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fa5]", s))


def collect_texts(obj: Any, parent_key: str = "", depth: int = 0) -> List[str]:
    if depth > 5 or obj is None:
        return []
    out: List[str] = []
    if isinstance(obj, str):
        if is_text_like(obj):
            out.append(re.sub(r"\s+", " ", obj).strip())
        return out
    if isinstance(obj, (int, float, bool)):
        return []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            # Always inspect likely text fields first.
            if any(h in lk for h in TEXT_KEY_HINTS):
                out.extend(collect_texts(v, lk, depth + 1))
        for k, v in obj.items():
            lk = str(k).lower()
            if any(h in lk for h in TEXT_KEY_HINTS):
                continue
            if isinstance(v, (dict, list, tuple)):
                out.extend(collect_texts(v, lk, depth + 1))
            elif depth <= 2:
                out.extend(collect_texts(v, lk, depth + 1))
        return out
    if isinstance(obj, (list, tuple)):
        for v in list(obj)[:20]:
            out.extend(collect_texts(v, parent_key, depth + 1))
        return out
    # pandas Series row support.
    if hasattr(obj, "to_dict"):
        try:
            return collect_texts(obj.to_dict(), parent_key, depth + 1)
        except Exception:
            return []
    return []


def best_context(obj: Any, max_chars: int = 420) -> str:
    seen = set()
    texts = []
    for t in collect_texts(obj):
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            texts.append(t)
    if not texts:
        return ""
    # Prefer concise captions/titles over long body text.
    texts.sort(key=lambda s: (0 if 20 <= len(s) <= 260 else 1, abs(len(s) - 120)))
    joined = " ".join(texts[:3])
    return re.sub(r"\s+", " ", joined).strip()[:max_chars]


def collect_ids(obj: Any, parent_key: str = "", depth: int = 0) -> List[str]:
    if depth > 5 or obj is None:
        return []
    out: List[str] = []
    if isinstance(obj, str):
        if any(h in parent_key.lower() for h in ID_KEY_HINTS):
            nid = norm_id(obj)
            if nid:
                out.append(nid)
        return out
    if isinstance(obj, (int, float)):
        if any(h in parent_key.lower() for h in ID_KEY_HINTS):
            out.append(norm_id(obj))
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(h in lk for h in ID_KEY_HINTS):
                if isinstance(v, (str, int, float)):
                    nid = norm_id(v)
                    if nid:
                        out.append(nid)
            if isinstance(v, (dict, list, tuple)):
                out.extend(collect_ids(v, lk, depth + 1))
        return out
    if isinstance(obj, (list, tuple)):
        for v in list(obj)[:30]:
            out.extend(collect_ids(v, parent_key, depth + 1))
        return out
    if hasattr(obj, "to_dict"):
        try:
            return collect_ids(obj.to_dict(), parent_key, depth + 1)
        except Exception:
            return []
    return out


def unique(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = norm_id(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def pickle_records(obj: Any) -> Iterator[Tuple[Any, Any]]:
    # dict: key -> record
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
        return
    # list: index -> record
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield i, v
        return
    # pandas DataFrame
    if hasattr(obj, "iterrows"):
        for i, row in obj.iterrows():
            yield i, row.to_dict()
        return


def source_from_file(path: Path) -> str:
    s = path.stem.lower()
    s = s.replace("processed_", "")
    if s in {"bbc_1", "bbc_2"}:
        return "bbc"
    return s


def build_metadata_index(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    files = []
    for pat in ["*.p", "*.pkl", "*.pickle"]:
        files.extend(metadata_dir.rglob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No metadata pickle files found under {metadata_dir}")

    index: Dict[str, Dict[str, Any]] = {}
    debug = []
    for path in files:
        try:
            obj = load_pickle(path)
        except Exception as exc:
            debug.append({"file": str(path), "error": str(exc)})
            continue
        n = 0
        kept = 0
        for key, val in pickle_records(obj):
            n += 1
            ids = unique([key] + collect_ids(val))
            ctx = best_context(val)
            if not ctx:
                continue
            kept += 1
            entry = {
                "context": ctx,
                "source": source_from_file(path),
                "metadata_file": str(path),
                "raw_key": str(key),
            }
            for idv in ids:
                index.setdefault(idv, entry)
        debug.append({"file": str(path), "records": n, "kept_with_context": kept})
    print(json.dumps({"metadata_files": debug, "index_size": len(index)}, ensure_ascii=False, indent=2), file=sys.stderr)
    return index


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


def iter_newsclippings_json(data_dir: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for path in sorted(data_dir.rglob("*.json")):
        records = load_json_records(path)
        for rec in records:
            yield path, rec


def get_first(rec: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in rec and rec[k] not in [None, ""]:
            return rec[k]
    return None


def derive_split_generator(path: Path) -> Tuple[str, str]:
    parts = [p.lower() for p in path.parts]
    name = path.stem.lower()
    joined = "_".join(parts + [name])
    split = "unknown"
    for s in ["train", "val", "test", "valid", "validation"]:
        if re.search(rf"(^|[_\-.\\/]){s}($|[_\-.\\/])", joined):
            split = "val" if s in {"valid", "validation"} else s
            break
    generator = name
    for s in ["train", "val", "test", "valid", "validation"]:
        generator = re.sub(rf"(^|[_\-.]){s}($|[_\-.])", "_", generator)
    return split, generator.strip("_.-") or name


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


def rec_text(rec: Dict[str, Any], meta: Dict[str, Dict[str, Any]], text_id: str) -> str:
    direct = best_context(rec)
    if direct and len(direct.split()) >= 3:
        return direct
    if text_id in meta:
        return meta[text_id]["context"]
    return direct


def maybe_sample(rows: List[Dict[str, Any]], max_records: int, seed: int) -> List[Dict[str, Any]]:
    if max_records <= 0 or len(rows) <= max_records:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newsclippings-data-dir", required=True)
    ap.add_argument("--visualnews-metadata-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--generator", default="")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    data_dir = Path(args.newsclippings_data_dir)
    meta_dir = Path(args.visualnews_metadata_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] scanning NewsCLIPpings json recursively from {data_dir}", file=sys.stderr)
    json_files = sorted(data_dir.rglob("*.json"))
    print(f"[INFO] json file count={len(json_files)}", file=sys.stderr)
    print(f"[INFO] loading VisualNews metadata from {meta_dir}", file=sys.stderr)
    meta = build_metadata_index(meta_dir)
    print(f"[INFO] metadata index size={len(meta)}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    stats = {"json_files": len(json_files), "total": 0, "kept": 0, "missing_ids": 0, "missing_text": 0, "missing_true_context": 0}

    for path, rec in iter_newsclippings_json(data_dir):
        split, generator = derive_split_generator(path)
        if args.split and split != args.split:
            continue
        if args.generator and args.generator.lower() not in generator.lower():
            continue
        stats["total"] += 1
        sample_id = norm_id(get_first(rec, TEXT_ID_KEYS))
        image_id = norm_id(get_first(rec, IMAGE_KEYS))
        text_id = sample_id
        label = normalize_label(get_first(rec, LABEL_KEYS))
        if not sample_id or not image_id:
            stats["missing_ids"] += 1
            continue
        current = rec_text(rec, meta, text_id)
        true_entry = meta.get(image_id, {})
        true_ctx = true_entry.get("context", "")
        if not current:
            stats["missing_text"] += 1
        if not true_ctx:
            stats["missing_true_context"] += 1
        if not current or not true_ctx:
            continue
        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "text_id": text_id,
            "split": split,
            "generator": generator,
            "domain": true_entry.get("source", "unknown"),
            "label": label,
            "current_caption": current,
            "true_image_context": true_ctx,
            "source": "visualnews_metadata",
            "newsclippings_file": str(path),
        })

    rows = maybe_sample(rows, args.max_records, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["kept"] = len(rows)
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "stats": stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
