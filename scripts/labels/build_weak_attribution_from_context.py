from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Build weak attribution labels from COVE-lite context pairs.")
    ap.add_argument("--input", required=True, help="outputs/cove_lite_context_pairs.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", default="event_sidecar_demo", help="Pipeline classification policy. Attribution is independent from final label.")
    args = ap.parse_args()

    pipe = E3VDTPipeline(classification_policy=args.policy)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    missing = 0
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
            row = dict(rec)
            row.update({
                "weak_label": pred.get("label"),
                "weak_mismatch_type": pred.get("mismatch_type"),
                "weak_conflict_fields": pred.get("conflict_fields", []),
                "event_scores": pred.get("event_scores", {}),
                "text_event": pred.get("text_event", {}),
                "image_event": pred.get("image_event", {}),
                "weak_explanation": pred.get("explanation", ""),
                "weak_method": "cove_lite_event_rule_v1",
            })
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    stats = {"input": args.input, "output": str(out_path), "records": n, "missing_text_or_context": missing}
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
