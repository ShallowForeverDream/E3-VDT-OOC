from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline


def main() -> int:
    cases_path = ROOT / "examples" / "demo_cases.jsonl"
    pipe = E3VDTPipeline()
    rows = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        result = pipe.predict(text=case["text"], image_context=case.get("image_context", "")).to_dict()
        ok = (
            result["label"] == case["expected_label"]
            and result["mismatch_type"] == case["expected_mismatch_type"]
            and result["conflict_fields"] == case["expected_conflict_fields"]
        )
        rows.append((case, result, ok))

    print("| ID | expected | actual | confidence | conflict_fields | status |")
    print("|---|---|---|---:|---|---|")
    for case, result, ok in rows:
        expected = f'{case["expected_label"]} / {case["expected_mismatch_type"]}'
        actual = f'{result["label"]} / {result["mismatch_type"]}'
        fields = ",".join(result["conflict_fields"]) or "-"
        print(f'| {case["id"]} | {expected} | {actual} | {result["confidence"]:.2f} | {fields} | {"OK" if ok else "CHECK"} |')

    failed = [case["id"] for case, _, ok in rows if not ok]
    if failed:
        print("\nFailed cases:", ", ".join(failed))
        return 1
    print(f"\n[OK] {len(rows)} demo cases matched current expected outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
