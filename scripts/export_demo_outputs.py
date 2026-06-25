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
from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline


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

    cove_cases_path = ROOT / "examples" / "cove_attr_demo_cases.jsonl"
    cove_rows = []
    if cove_cases_path.exists():
        cove_pipe = VDTCOVEAttrPipeline()
        for line in cove_cases_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            cove_rows.append({
                "input": case,
                "actual": cove_pipe.predict(
                    current_caption=case.get("current_caption", ""),
                    true_image_context=case.get("true_image_context", ""),
                    vdt_label=case.get("vdt_label"),
                    vdt_score=case.get("vdt_score"),
                    sample_id=case.get("sample_id", ""),
                    image_id=case.get("image_id", ""),
                    domain=case.get("domain", "demo"),
                ),
            })
        cove_out = ROOT / "examples" / "cove_attr_demo_outputs_full.json"
        cove_out.write_text(json.dumps(cove_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(cove_out)
    print(out_path)
    print(f"[OK] exported {len(rows)} legacy demo outputs and {len(cove_rows)} VDT-COVE-Attr outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
