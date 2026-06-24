from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from e3vdt.inference.pipeline import E3VDTPipeline
pipe=E3VDTPipeline()
result=pipe.predict_dict(text="A protest erupted in Paris on Monday.", image_context="People gathered in London during an earlier climate demonstration in 2020.")
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result["label"] in {"OOC","Non-OOC","Uncertain"}
assert "mismatch_type" in result and "event_scores" in result
print("\n[OK] E3-VDT-OOC project self-check passed.")
