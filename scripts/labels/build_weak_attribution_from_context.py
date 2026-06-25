from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline


def norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())


def safeguard(rec: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    pred = dict(pred)
    label = rec.get("label")
    cur = norm_text(rec.get("current_caption", ""))
    true = norm_text(rec.get("true_image_context", ""))
    if label == 0 and cur and cur == true:
        pred["label"] = "Non-OOC"
        pred["mismatch_type"] = "benign illustrative image"
        pred["conflict_fields"] = []
        pred["explanation"] = "Non-OOC pair: current caption equals true image context."
    elif label == 1 and cur and true and cur != true and not pred.get("conflict_fields"):
        pred["mismatch_type"] = "uncertain / evidence insufficient"
        pred["conflict_fields"] = ["evidence_insufficient"]
        pred["explanation"] = "OOC pair but extracted fields are too sparse for reliable attribution."
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Build weak attribution labels from COVE-lite context pairs.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", default="event_sidecar_demo")
    args = ap.parse_args()

    pipe = E3VDTPipeline(classification_policy=args.policy)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    missing = 0
    type_counts: Dict[str, int] = {}
    with open(args.input, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec: Dict[str, Any] = json.loads(line)
            text = rec.get("current_caption") or rec.get("text") or ""
            true_ctx = rec.get("true_image_context") or rec.get("image_context") or ""
            if not text or not true_ctx:
                missing += 1
            pred = pipe.predict_dict(text=text, image_context=true_ctx, classification_policy=args.policy)
            pred = safeguard(rec, pred)
            row = dict(rec)
            row.update({
                "weak_label": pred.get("label"),
                "weak_mismatch_type": pred.get("mismatch_type"),
                "weak_conflict_fields": pred.get("conflict_fields", []),
                "event_scores": pred.get("event_scores", {}),
                "text_event": pred.get("text_event", {}),
                "image_event": pred.get("image_event", {}),
                "weak_explanation": pred.get("explanation", ""),
                "weak_method": "cove_lite_event_rule_v2_label_aware",
            })
            type_counts[row["weak_mismatch_type"]] = type_counts.get(row["weak_mismatch_type"], 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    stats = {"input": args.input, "output": str(out_path), "records": n, "missing_text_or_context": missing, "weak_mismatch_type_counts": type_counts}
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
