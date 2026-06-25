from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from scripts.features.build_image_caption_attribution_features import (  # noqa: E402
    FIELDS,
    ClipScorer,
    build_base_feature,
)

_SCORER_CACHE: Dict[Tuple[str, str, bool], ClipScorer] = {}


TYPE_TO_FIELDS = {
    "benign illustrative image": [],
    "none": [],
    "entity mismatch": ["entity"],
    "location mismatch": ["location"],
    "temporal mismatch": ["time"],
    "event-type mismatch": ["event_type"],
    "relation mismatch": ["relation"],
    "different-event mismatch": ["entity", "location", "event_type"],
    "uncertain / evidence insufficient": ["evidence_insufficient"],
}


def as_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def load_image(image_path: str):
    if Image is None or not image_path:
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def enrich_clip_scores(feat: Dict[str, Any], prompts: List[str], image_path: str, clip_model: str, device: str, no_clip: bool) -> Dict[str, Any]:
    img = load_image(image_path)
    feat["image_loaded"] = int(img is not None)
    vals = [0.0] * 6
    clip_enabled = False
    if img is not None:
        key = (clip_model, device, bool(no_clip))
        scorer = _SCORER_CACHE.get(key)
        if scorer is None:
            scorer = ClipScorer(clip_model, device, no_clip=no_clip)
            _SCORER_CACHE[key] = scorer
        clip_enabled = scorer.enabled
        vals = scorer.score_batch([img], [prompts])[0]
    feat["clip_caption_sim"] = vals[0] if vals else 0.0
    for j, field in enumerate(FIELDS, start=1):
        feat[f"clip_prompt_sim_{field}"] = vals[j] if j < len(vals) else 0.0
    prompt_vals = [feat[f"clip_prompt_sim_{f}"] for f in FIELDS if feat.get(f"{f}_present")]
    feat["clip_prompt_sim_min_present"] = min(prompt_vals) if prompt_vals else 0.0
    feat["clip_prompt_sim_mean_present"] = sum(prompt_vals) / len(prompt_vals) if prompt_vals else 0.0
    feat["clip_prompt_sim_max_present"] = max(prompt_vals) if prompt_vals else 0.0
    feat["clip_enabled"] = clip_enabled
    return feat


def build_feature_row(image_path: str, caption: str, vdt_score: float, clip_model: str, device: str, no_clip: bool) -> Dict[str, Any]:
    row = {
        "sample_id": "manual",
        "image_id": Path(image_path).stem if image_path else "manual-image",
        "current_caption": caption,
        "vdt_score": vdt_score,
        "gold_mismatch_type": "",
        "gold_conflict_fields": [],
    }
    feat, prompts = build_base_feature(row, image_path=image_path, image_loaded=False)
    return enrich_clip_scores(feat, prompts, image_path=image_path, clip_model=clip_model, device=device, no_clip=no_clip)


def matrix_from_feature(feat: Dict[str, Any], feature_names: Sequence[str]) -> List[List[float]]:
    return [[as_float(feat.get(k)) for k in feature_names]]


def prompt_rule(feat: Dict[str, Any]) -> Tuple[str, Set[str], float]:
    sims = {
        f: as_float(feat.get(f"clip_prompt_sim_{f}"))
        for f in FIELDS
        if as_float(feat.get(f"{f}_present")) > 0
    }
    if as_float(feat.get("image_loaded")) <= 0:
        return "uncertain / evidence insufficient", {"evidence_insufficient"}, 0.0
    if not sims:
        return "uncertain / evidence insufficient", {"evidence_insufficient"}, 0.2
    field = min(sims, key=sims.get)
    spread = max(sims.values()) - min(sims.values())
    if spread < 0.015:
        return "benign illustrative image", set(), max(0.35, 1.0 - spread)
    typ = {
        "entity": "entity mismatch",
        "location": "location mismatch",
        "time": "temporal mismatch",
        "event_type": "event-type mismatch",
        "relation": "relation mismatch",
    }[field]
    return typ, {field}, min(0.95, max(0.35, spread * 8))


def predict_with_model(feat: Dict[str, Any], model_path: Path) -> Optional[Tuple[str, Set[str], float, Dict[str, Any]]]:
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    feature_names = list(bundle.get("feature_names") or [])
    if not feature_names:
        return None
    X = matrix_from_feature(feat, feature_names)
    type_model = bundle["type_model"]
    type_encoder = bundle["type_encoder"]
    pred_id = type_model.predict(X)[0]
    mismatch_type = str(type_encoder.inverse_transform([pred_id])[0])
    confidence = 0.0
    if hasattr(type_model, "predict_proba"):
        try:
            probs = type_model.predict_proba(X)[0]
            confidence = float(max(probs))
        except Exception:
            confidence = 0.0
    field_model = bundle["field_model"]
    fields = list(bundle.get("fields") or FIELDS)
    arr = field_model.predict(X)[0]
    conflict_fields = {fields[i] for i, v in enumerate(list(arr)[: len(fields)]) if int(v)}
    if not conflict_fields:
        conflict_fields = set(TYPE_TO_FIELDS.get(mismatch_type, []))
    meta = {
        "model_path": str(model_path),
        "feature_names": feature_names,
        "head_kind": bundle.get("head_kind", ""),
        "feature_groups": bundle.get("feature_groups", []),
    }
    return mismatch_type, conflict_fields, confidence, meta


def predict(
    image_path: str,
    caption: str,
    vdt_label: str = "OOC",
    vdt_score: float = 0.87,
    model_path: str = "outputs/no_true_context_attr/no_true_context_attr_head.pkl",
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    no_clip: bool = False,
) -> Dict[str, Any]:
    feat = build_feature_row(image_path, caption, vdt_score=vdt_score, clip_model=clip_model, device=device, no_clip=no_clip)
    if str(vdt_label).strip().lower() in {"non-ooc", "non_ooc", "non ooc", "benign"}:
        mismatch_type = "benign illustrative image"
        conflict_fields: Set[str] = set()
        confidence = float(vdt_score)
        source = "vdt_non_ooc_gate"
        model_meta: Dict[str, Any] = {}
    else:
        pred = predict_with_model(feat, Path(model_path))
        if pred is None:
            mismatch_type, conflict_fields, confidence = prompt_rule(feat)
            source = "field_prompt_grounding_rule_fallback"
            model_meta = {"model_path": model_path, "model_loaded": False}
        else:
            mismatch_type, conflict_fields, confidence, model_meta = pred
            source = "no_true_context_attr_head"
            model_meta["model_loaded"] = True

    evidence_status = "visually_grounded" if feat.get("image_loaded") and feat.get("clip_enabled") else "uncertain"
    if mismatch_type == "benign illustrative image":
        explanation = "VDT-CF-Attr 未发现明确字段错配，或 VDT 主分类为 Non-OOC。"
    elif mismatch_type == "uncertain / evidence insufficient":
        explanation = "图片或视觉语言特征不足，系统不强行解释具体错配类型。"
    else:
        explanation = f"VDT 判断为 {vdt_label}；VDT-CF-Attr 在不使用 true context 的条件下预测主要错配类型为 {mismatch_type}，冲突字段为 {', '.join(sorted(conflict_fields)) or 'unknown'}。"

    return {
        "vdt_label": vdt_label,
        "vdt_score": round(float(vdt_score), 4),
        "mismatch_type": mismatch_type,
        "conflict_fields": sorted(conflict_fields),
        "confidence": round(float(confidence), 4),
        "evidence_status": evidence_status,
        "uses_true_context": False,
        "decision_source": source,
        "caption": caption,
        "image_path": image_path,
        "feature_summary": {
            "clip_enabled": bool(feat.get("clip_enabled")),
            "image_loaded": bool(feat.get("image_loaded")),
            "clip_caption_sim": round(as_float(feat.get("clip_caption_sim")), 4),
            "prompt_sims": {f: round(as_float(feat.get(f"clip_prompt_sim_{f}")), 4) for f in FIELDS},
            "field_presence": {f: int(as_float(feat.get(f"{f}_present"))) for f in FIELDS},
        },
        "model": model_meta,
        "explanation": explanation,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer VDT-CF-Attr without true context.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--vdt-label", default="OOC")
    ap.add_argument("--vdt-score", type=float, default=0.87)
    ap.add_argument("--model", default="outputs/no_true_context_attr/no_true_context_attr_head.pkl")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()
    obj = predict(
        image_path=args.image,
        caption=args.caption,
        vdt_label=args.vdt_label,
        vdt_score=args.vdt_score,
        model_path=args.model,
        clip_model=args.clip_model,
        device=args.device,
        no_clip=args.no_clip,
    )
    print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
