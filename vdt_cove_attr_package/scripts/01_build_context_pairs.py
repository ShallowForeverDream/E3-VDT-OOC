from __future__ import annotations
import argparse, json, pickle, re, os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from tqdm import tqdm

TEXT_KEYS = ["caption", "current_caption", "text", "sentence", "title", "article_title", "headline", "description", "context"]
CONTEXT_KEYS = ["caption", "title", "article_title", "headline", "description", "text", "context", "article"]
ID_KEYS = ["image_id", "imageId", "img_id", "photo_id", "id", "uid", "nid", "article_id", "text_id", "caption_id"]
SAMPLE_ID_KEYS = ["sample_id", "id", "pair_id", "uid"]
LABEL_KEYS = ["label", "falsified", "is_falsified", "is_ooc", "target"]


def norm_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    return s


def pick_text(rec: Dict[str, Any], keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    # Sometimes metadata stores a list of captions/tokens.
    for k in ["captions", "sentences"]:
        v = rec.get(k)
        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                parts.append(" ".join(vals[:3]))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def iter_records(obj: Any, parent_key: str = "") -> Iterator[Dict[str, Any]]:
    # pandas DataFrame
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        try:
            for r in obj.to_dict(orient="records"):
                if isinstance(r, dict):
                    yield r
            return
        except Exception:
            pass
    if isinstance(obj, dict):
        # dict of columns, same length
        if obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
            lens = [len(v) for v in obj.values() if isinstance(v, (list, tuple))]
            if lens and len(set(lens)) == 1 and lens[0] > 0:
                keys = list(obj.keys())
                for i in range(lens[0]):
                    yield {k: obj[k][i] for k in keys}
                return
        # dict id -> record
        for k, v in obj.items():
            if isinstance(v, dict):
                r = dict(v)
                r.setdefault("_dict_key", k)
                yield r
            elif isinstance(v, (list, tuple, dict)):
                yield from iter_records(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            if isinstance(x, dict):
                yield x
            else:
                yield from iter_records(x, parent_key)


def collect_ids(rec: Dict[str, Any]) -> List[str]:
    ids = []
    for k in ID_KEYS + ["_dict_key", "image_url", "url", "path", "filename", "file_name"]:
        v = rec.get(k)
        if isinstance(v, (list, tuple)):
            vals = v
        else:
            vals = [v]
        for item in vals:
            sid = norm_id(item)
            if sid:
                ids.append(sid)
                # add basename/stem variants for URLs/paths
                base = os.path.basename(sid)
                if base and base != sid:
                    ids.append(base)
                    ids.append(os.path.splitext(base)[0])
    out=[]; seen=set()
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_metadata_index(visual_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], dict]:
    files=[]
    for pat in ["processed_*.p", "*.pkl", "*.pickle", "*.p"]:
        files.extend(sorted(visual_dir.rglob(pat)))
    files = list(dict.fromkeys(files))
    index: Dict[str, Dict[str, Any]] = {}
    file_stats=[]
    for p in files:
        records=0; kept=0; errors=[]
        try:
            obj = load_pickle(p)
            for rec in iter_records(obj):
                records += 1
                if not isinstance(rec, dict):
                    continue
                ctx = pick_text(rec, CONTEXT_KEYS)
                if not ctx:
                    continue
                ids = collect_ids(rec)
                if not ids:
                    continue
                meta = {"true_image_context": ctx, "metadata_file": str(p), "raw_ids": ids[:20]}
                for sid in ids:
                    index.setdefault(sid, meta)
                kept += 1
        except Exception as e:
            errors.append(repr(e))
        file_stats.append({"file": str(p), "records": records, "kept_with_context": kept, "errors": errors[:3]})
    stats={"metadata_files": file_stats, "index_size": len(index)}
    return index, stats


def load_json_records(news_dir: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for p in sorted(list(news_dir.rglob("*.json")) + list(news_dir.rglob("*.jsonl"))):
        try:
            if p.suffix.lower() == ".jsonl":
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj=json.loads(line)
                            for rec in iter_records(obj):
                                yield p, rec
            else:
                obj=json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                for rec in iter_records(obj):
                    yield p, rec
        except Exception:
            continue


def infer_label(rec: Dict[str, Any]) -> Optional[int]:
    for k in LABEL_KEYS:
        if k in rec:
            v=rec[k]
            if isinstance(v, bool): return int(v)
            if isinstance(v, (int,float)): return int(v)
            s=str(v).lower()
            if s in {"1","true","fake","falsified","ooc","misleading"}: return 1
            if s in {"0","false","real","non-ooc","non_ooc","match"}: return 0
    return None


def main():
    ap=argparse.ArgumentParser(description="Build COVE-lite context pairs from NewsCLIPpings and VisualNews metadata.")
    ap.add_argument("--news-dir", required=True)
    ap.add_argument("--visual-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args=ap.parse_args()
    news_dir=Path(args.news_dir); visual_dir=Path(args.visual_dir)
    index, meta_stats = build_metadata_index(visual_dir)
    print(json.dumps({"metadata": meta_stats}, indent=2, ensure_ascii=False)[:4000])
    out_path=Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    stats={"total":0,"kept":0,"missing_ids":0,"missing_text":0,"missing_true_context":0,"label_missing":0,"metadata_index_size":len(index)}
    with open(out_path,"w",encoding="utf-8") as out:
        for p, rec in tqdm(load_json_records(news_dir), desc="news records"):
            stats["total"] += 1
            if args.max_records and stats["total"] > args.max_records:
                break
            ids = collect_ids(rec)
            if not ids:
                stats["missing_ids"] += 1
                continue
            sample_id = next((norm_id(rec.get(k)) for k in SAMPLE_ID_KEYS if norm_id(rec.get(k))), None) or f"{p.stem}:{stats['total']}"
            image_id = next((norm_id(rec.get(k)) for k in ["image_id","imageId","img_id","photo_id"] if norm_id(rec.get(k))), None) or ids[0]
            text_id = next((norm_id(rec.get(k)) for k in ["text_id","caption_id","article_id","id"] if norm_id(rec.get(k))), None)
            current_caption = pick_text(rec, TEXT_KEYS)
            if not current_caption and text_id and text_id in index:
                current_caption = index[text_id]["true_image_context"]
            if not current_caption:
                stats["missing_text"] += 1
                continue
            true_meta = None
            for sid in [image_id] + ids:
                if sid in index:
                    true_meta = index[sid]; break
            if not true_meta:
                stats["missing_true_context"] += 1
                continue
            label = infer_label(rec)
            if label is None:
                stats["label_missing"] += 1
            row={
                "sample_id": sample_id,
                "image_id": image_id,
                "text_id": text_id,
                "split": rec.get("split") or p.stem,
                "domain": rec.get("domain") or rec.get("source") or rec.get("news_source"),
                "generator": rec.get("generator") or rec.get("method"),
                "label": label,
                "current_caption": current_caption,
                "true_image_context": true_meta["true_image_context"],
                "true_context_source": true_meta.get("metadata_file"),
                "source_file": str(p),
            }
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
            stats["kept"] += 1
    stats["coverage"] = stats["kept"] / stats["total"] if stats["total"] else 0.0
    stats["metadata"] = meta_stats
    stats_path=Path(args.stats or str(out_path)+".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
