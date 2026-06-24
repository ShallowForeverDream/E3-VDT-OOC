from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline


def load_cases() -> list[dict]:
    path = ROOT / "examples" / "demo_cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export deterministic demo outputs for classroom fallback presentation.")
    parser.add_argument("--out", default=str(ROOT / "examples" / "demo_outputs.json"), help="Output JSON path.")
    args = parser.parse_args()

    pipe = E3VDTPipeline()
    rows = []
    for case in load_cases():
        result = pipe.predict_dict(text=case["text"], image_context=case.get("image_context", ""))
        rows.append(
            {
                "id": case["id"],
                "scenario": case.get("scenario", ""),
                "demo_point": case.get("demo_point", ""),
                "text": case["text"],
                "image_context": case.get("image_context", ""),
                "expected": {
                    "label": case["expected_label"],
                    "mismatch_type": case["expected_mismatch_type"],
                    "conflict_fields": case["expected_conflict_fields"],
                },
                "actual": result,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(f"[OK] exported {len(rows)} demo outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
