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
    """Verify that attribution fields cannot override VDT baseline labels.

    This is a project-level guardrail for the final design requirement:
    classification accuracy may stay equal to VDT, but must not be reduced by
    the explanation/mismatch sidecar.
    """
    pipe = E3VDTPipeline(classification_policy="baseline_preserving")
    adversarial_cases = [
        {
            "text": "A protest erupted in Paris on Monday.",
            "image_context": "People gathered in London in 2020.",
            "baseline_label": "Non-OOC",
            "baseline_score": 0.91,
        },
        {
            "text": "A flood caused evacuations in Shanghai in 2024.",
            "image_context": "A flood caused evacuations in Shanghai in 2024.",
            "baseline_label": "OOC",
            "baseline_score": 0.88,
        },
    ]
    rows = []
    ok = True
    for case in adversarial_cases:
        result = pipe.predict(**case).to_dict()
        same = result["label"] == case["baseline_label"]
        rows.append(
            {
                "baseline_label": case["baseline_label"],
                "result_label": result["label"],
                "decision_source": result["decision_source"],
                "mismatch_type": result["mismatch_type"],
                "ok": same,
            }
        )
        ok = ok and same and result["decision_source"] == "vdt_baseline"

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if not ok:
        print("[FAIL] baseline-preserving guardrail was violated.")
        return 1
    print("[OK] baseline-preserving mode keeps final labels identical to VDT baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
