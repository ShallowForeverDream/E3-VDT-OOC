from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from e3vdt.event.extractor import extract_event_tuple  # noqa: E402


FIELDS = ["entity", "location", "time", "event_type", "relation"]
FIELD_ATTR = {
    "entity": "entities",
    "location": "locations",
    "time": "times",
    "event_type": "event_types",
    "relation": "relations",
}
TYPE_TO_FIELDS = {
    "benign illustrative image": [],
    "none": [],
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "different-event mismatch": ["entity", "location", "event_type"],
    "global/uncontrolled mismatch": ["entity", "location", "event_type"],
    "uncertain / evidence insufficient": ["evidence_insufficient"],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_id(x: Any) -> str:
    return str(x or "").strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def field_set(xs: Any, mismatch_type: str = "") -> List[str]:
    if isinstance(xs, str):
        parts = [x.strip() for x in xs.replace(";", ",").split(",")]
    else:
        parts = [str(x).strip() for x in (xs or [])]
    out = [x for x in parts if x]
    if not out:
        out = TYPE_TO_FIELDS.get(mismatch_type, [])
    return out


def make_prompt(field: str, values: Sequence[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return ""
    text = ", ".join(vals[:3])
    if field == "entity":
        return f"a news photo involving {text}"
    if field == "location":
        return f"a news photo taken in {text}"
    if field == "time":
        return f"a news photo from {text}"
    if field == "event_type":
        return f"a news photo about {text}"
    if field == "relation":
        return f"a news photo showing {text}"
    return f"a news photo about {text}"


def load_origin_image_index(origin_data_json: Optional[Path], needed_ids: set[str]) -> Dict[str, str]:
    if not origin_data_json or not origin_data_json.exists():
        return {}
    obj = json.loads(origin_data_json.read_text(encoding="utf-8"))
    records = obj if isinstance(obj, list) else list(obj.values()) if isinstance(obj, dict) else []
    out: Dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = clean_id(rec.get("id"))
        if rid in needed_ids and rec.get("image_path"):
            out[rid] = str(rec["image_path"])
            if len(out) >= len(needed_ids):
                break
    return out


def member_from_image_path(image_path: str) -> str:
    s = str(image_path or "").replace("\\", "/").strip()
    s = s[2:] if s.startswith("./") else s
    if s.startswith("origin/"):
        return s
    return "origin/" + s


def load_tar_index(path: Optional[Path]) -> Dict[str, Dict[str, int]]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ImageLoader:
    image_root: Optional[Path] = None
    origin_tar: Optional[Path] = None
    tar_index: Optional[Dict[str, Dict[str, int]]] = None

    def load(self, image_path: str):
        if Image is None:
            return None
        member = member_from_image_path(image_path)
        local_candidates: List[Path] = []
        if self.image_root:
            raw = image_path.replace("\\", "/")
            raw = raw[2:] if raw.startswith("./") else raw
            local_candidates.extend([
                self.image_root / raw,
                self.image_root / member,
                self.image_root / member.replace("origin/", "", 1),
            ])
        for p in local_candidates:
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
        if self.origin_tar and self.tar_index and member in self.tar_index:
            meta = self.tar_index[member]
            with self.origin_tar.open("rb") as f:
                f.seek(int(meta["offset"]))
                data = f.read(int(meta["size"]))
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None
        return None


class ClipScorer:
    def __init__(self, model_name: str, device: str, no_clip: bool = False) -> None:
        self.enabled = False
        self.model = None
        self.processor = None
        self.device = device
        if no_clip:
            return
        if torch is None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] CLIP unavailable, using zero similarities: {exc}", file=sys.stderr)

    def score_batch(self, images: List[Any], text_groups: List[List[str]]) -> List[List[float]]:
        if not self.enabled or not images:
            return [[0.0 for _ in group] for group in text_groups]
        flat_texts = [t if t else "empty" for group in text_groups for t in group]
        with torch.no_grad():
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            text_inputs = self.processor(text=flat_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = self._as_tensor(image_features)
            text_features = self._as_tensor(text_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out: List[List[float]] = []
            pos = 0
            for i, group in enumerate(text_groups):
                vals = []
                for _ in group:
                    vals.append(float((image_features[i] * text_features[pos]).sum().detach().cpu()))
                    pos += 1
                out.append(vals)
            return out

    @staticmethod
    def _as_tensor(x: Any):
        if hasattr(x, "image_embeds"):
            return x.image_embeds
        if hasattr(x, "text_embeds"):
            return x.text_embeds
        if hasattr(x, "pooler_output"):
            return x.pooler_output
        if isinstance(x, (tuple, list)):
            return x[0]
        return x


def build_base_feature(row: Dict[str, Any], image_path: str, image_loaded: bool) -> Tuple[Dict[str, Any], List[str]]:
    caption = str(row.get("current_caption") or row.get("caption") or "")
    event = extract_event_tuple(caption, source="current_caption")
    prompts: List[str] = [caption]
    feat: Dict[str, Any] = {
        "sample_id": row.get("sample_id", ""),
        "source_sample_id": row.get("source_sample_id", ""),
        "image_id": row.get("image_id", ""),
        "domain": row.get("domain", ""),
        "split": row.get("split", ""),
        "current_caption": caption,
        "image_path": image_path,
        "image_loaded": int(image_loaded),
        "vdt_score": float(row.get("vdt_score") or 0.5),
        "caption_chars": len(caption),
        "caption_tokens": token_count(caption),
        "gold_mismatch_type": row.get("gold_mismatch_type", ""),
    }
    gold_fields = set(field_set(row.get("gold_conflict_fields"), str(row.get("gold_mismatch_type") or "")))
    for field in FIELDS:
        vals = list(getattr(event, FIELD_ATTR[field]))
        feat[f"{field}_count"] = len(vals)
        feat[f"{field}_present"] = int(bool(vals))
        feat[f"gold_field_{field}"] = int(field in gold_fields)
        prompt = make_prompt(field, vals)
        prompts.append(prompt)
        feat[f"{field}_prompt"] = prompt
    return feat, prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build no-true-context image+caption attribution features.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--origin-data-json", default="E:/OOC_Datasets/VisualNews/origin/data.json")
    ap.add_argument("--origin-tar", default="E:/OOC_Datasets/VisualNews/origin.tar")
    ap.add_argument("--tar-index", default="D:/MY_PROJECT/OOC/datasets/visualnews_train_test_tar_index.json")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda" if torch is not None and getattr(torch.cuda, "is_available", lambda: False)() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    needed_ids = {clean_id(r.get("image_id")) for r in rows if clean_id(r.get("image_id"))}
    origin_map = load_origin_image_index(Path(args.origin_data_json) if args.origin_data_json else None, needed_ids)
    tar_index = load_tar_index(Path(args.tar_index) if args.tar_index else None)
    loader = ImageLoader(
        image_root=Path(args.image_root) if args.image_root else None,
        origin_tar=Path(args.origin_tar) if args.origin_tar else None,
        tar_index=tar_index,
    )
    scorer = ClipScorer(args.clip_model, args.device, no_clip=args.no_clip)

    out_rows: List[Dict[str, Any]] = []
    missing_image = 0
    used_image = 0
    for start in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[start : start + max(1, args.batch_size)]
        feats: List[Dict[str, Any]] = []
        prompts: List[List[str]] = []
        images: List[Any] = []
        image_positions: List[int] = []
        for row in batch:
            image_id = clean_id(row.get("image_id"))
            image_path = str(row.get("image_path") or origin_map.get(image_id) or "")
            img = loader.load(image_path) if image_path else None
            feat, text_group = build_base_feature(row, image_path=image_path, image_loaded=img is not None)
            feats.append(feat)
            prompts.append(text_group)
            if img is None:
                missing_image += 1
            else:
                used_image += 1
                images.append(img)
                image_positions.append(len(feats) - 1)
        scores_by_pos: Dict[int, List[float]] = {}
        if images:
            img_prompts = [prompts[i] for i in image_positions]
            for pos, vals in zip(image_positions, scorer.score_batch(images, img_prompts)):
                scores_by_pos[pos] = vals
        for i, feat in enumerate(feats):
            vals = scores_by_pos.get(i, [0.0] * 6)
            feat["clip_caption_sim"] = vals[0]
            for j, field in enumerate(FIELDS, start=1):
                feat[f"clip_prompt_sim_{field}"] = vals[j] if j < len(vals) else 0.0
            prompt_vals = [feat[f"clip_prompt_sim_{f}"] for f in FIELDS if feat.get(f"{f}_present")]
            feat["clip_prompt_sim_min_present"] = min(prompt_vals) if prompt_vals else 0.0
            feat["clip_prompt_sim_mean_present"] = sum(prompt_vals) / len(prompt_vals) if prompt_vals else 0.0
            feat["clip_prompt_sim_max_present"] = max(prompt_vals) if prompt_vals else 0.0
            out_rows.append(feat)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    stats = {
        "input": args.input,
        "output": args.output,
        "records": len(out_rows),
        "used_true_context_features": False,
        "clip_enabled": scorer.enabled,
        "clip_model": args.clip_model if scorer.enabled else None,
        "images_loaded": used_image,
        "images_missing": missing_image,
        "origin_data_json": args.origin_data_json,
        "origin_tar": args.origin_tar,
        "tar_index": args.tar_index,
    }
    out.with_suffix(out.suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
