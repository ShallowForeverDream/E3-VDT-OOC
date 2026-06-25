from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.features.build_image_caption_attribution_features import (  # noqa: E402
    ImageLoader,
    load_origin_image_index,
    load_tar_index,
)
from scripts.infer.infer_vdt_cf_attr import predict  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120] or "case"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export local demo cases for the no-true-context VDT-CF-Attr tab.")
    ap.add_argument("--input", default="outputs/no_true_context_attr/controlled_counterfactual_test.jsonl")
    ap.add_argument("--output", default="outputs/no_true_context_attr_demo_cases.jsonl")
    ap.add_argument("--image-dir", default="outputs/no_true_context_attr_demo_images")
    ap.add_argument("--model", default="outputs/no_true_context_attr/no_true_context_attr_head.pkl")
    ap.add_argument("--origin-data-json", default="E:/OOC_Datasets/VisualNews/origin/data.json")
    ap.add_argument("--origin-tar", default="E:/OOC_Datasets/VisualNews/origin.tar")
    ap.add_argument("--tar-index", default="D:/MY_PROJECT/OOC/datasets/visualnews_train_test_tar_index.json")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--allow-incorrect", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    needed = {str(r.get("image_id") or "").strip() for r in rows if str(r.get("image_id") or "").strip()}
    origin_map = load_origin_image_index(Path(args.origin_data_json), needed)
    loader = ImageLoader(origin_tar=Path(args.origin_tar), tar_index=load_tar_index(Path(args.tar_index)))
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        image_path = origin_map.get(image_id, "")
        img = loader.load(image_path) if image_path else None
        if img is None:
            continue
        out_img = image_dir / f"{safe_name(str(row.get('sample_id') or image_id))}.jpg"
        img.save(out_img, quality=90)
        obj = predict(
            image_path=str(out_img),
            caption=str(row.get("current_caption") or ""),
            vdt_label="OOC" if row.get("label") else "Non-OOC",
            vdt_score=0.87,
            model_path=args.model,
            clip_model=args.clip_model,
            device=args.device,
        )
        gold = str(row.get("gold_mismatch_type") or "")
        ok = obj["mismatch_type"] == gold or (gold == "benign illustrative image" and obj["mismatch_type"] in {"benign illustrative image", "none"})
        if ok or args.allow_incorrect:
            selected_by_type[gold or "unknown"].append({
                "sample_id": row.get("sample_id"),
                "image": str(out_img),
                "caption": row.get("current_caption"),
                "vdt_label": "OOC" if row.get("label") else "Non-OOC",
                "vdt_score": 0.87,
                "gold_mismatch_type": gold,
                "pred_mismatch_type": obj["mismatch_type"],
                "pred_conflict_fields": obj["conflict_fields"],
                "demo_note": "local VisualNews counterfactual demo; no true_context is provided to inference",
            })
    selected: List[Dict[str, Any]] = []
    preferred = ["entity mismatch", "location mismatch", "temporal mismatch", "different-event mismatch", "benign illustrative image"]
    while len(selected) < args.n and any(selected_by_type.values()):
        for typ in preferred + sorted(k for k in selected_by_type if k not in preferred):
            if selected_by_type.get(typ) and len(selected) < args.n:
                selected.append(selected_by_type[typ].pop(0))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "image_dir": str(image_dir), "records": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
