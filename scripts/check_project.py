from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.pipeline import E3VDTPipeline
from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline

pipe = E3VDTPipeline()
result = pipe.predict_dict(
    text="A protest erupted in Paris on Monday.",
    image_context="People gathered in London during an earlier climate demonstration in 2020.",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result["label"] in {"OOC", "Non-OOC", "Uncertain"}
assert "mismatch_type" in result and "event_scores" in result

guarded = E3VDTPipeline(classification_policy="baseline_preserving").predict_dict(
    text="A protest erupted in Paris on Monday.",
    image_context="People gathered in London during an earlier climate demonstration in 2020.",
    baseline_label="Non-OOC",
    baseline_score=0.93,
)
assert guarded["label"] == "Non-OOC"
assert guarded["decision_source"] == "vdt_baseline"

cove = VDTCOVEAttrPipeline().predict(
    current_caption="Protesters marched in Paris on Monday after a new climate policy.",
    true_image_context="Demonstrators gathered in London in 2019 during a climate protest.",
    vdt_label="OOC",
    vdt_score=0.91,
    sample_id="check",
    image_id="check-image",
)
assert cove["final_label"] == "OOC"
assert cove["decision_source"] == "vdt_baseline"
assert "location" in cove["conflict_fields"]
assert cove["evidence_relevance"]["sufficient"] is True
print("\n[OK] E3-VDT-OOC project self-check passed, including VDT-COVE-Attr system route.")
