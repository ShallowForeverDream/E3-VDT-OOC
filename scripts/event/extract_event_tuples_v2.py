from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.event.enhanced_extractor import event_from_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract event tuples from COVE-lite context pairs.")
    ap.add_argument("--input", required=True, help="JSONL with current_caption and true_image_context")
    ap.add_argument("--output", required=True)
    ap.add_argument("--extractor", default="enhanced", choices=["rule", "enhanced", "spacy"])
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    missing = 0
    field_counts = Counter()
    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec: Dict[str, Any] = json.loads(line)
            cur = rec.get("current_caption") or rec.get("text") or ""
            ctx = rec.get("true_image_context") or rec.get("image_context") or ""
            if not cur or not ctx:
                missing += 1
            cur_event = event_from_text(cur, extractor=args.extractor, spacy_model=args.spacy_model)
            true_event = event_from_text(ctx, extractor=args.extractor, spacy_model=args.spacy_model)
            for prefix, ev in [("current", cur_event), ("true", true_event)]:
                for k in ["entities", "locations", "times", "event_types", "relations"]:
                    if ev.get(k):
                        field_counts[f"{prefix}_{k}"] += 1
            row = dict(rec)
            row.update({
                "current_event": cur_event,
                "true_event": true_event,
                "event_extractor": args.extractor,
            })
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    stats = {"input": str(in_path), "output": str(out_path), "records": n, "missing_text_or_context": missing, "field_presence_counts": dict(field_counts), "extractor": args.extractor}
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
