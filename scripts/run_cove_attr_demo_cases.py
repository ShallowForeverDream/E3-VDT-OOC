from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline

FIELDS = ["entity", "location", "time", "event_type", "relation", "evidence_insufficient"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def norm_fields(xs: Any) -> Set[str]:
    if not xs:
        return set()
    return {str(x).strip() for x in xs}


def main() -> int:
    input_path = ROOT / "examples" / "cove_attr_demo_cases.jsonl"
    out_path = ROOT / "examples" / "cove_attr_demo_outputs.json"
    rows = read_jsonl(input_path)
    pipe = VDTCOVEAttrPipeline()
    outputs = []
    type_ok = exact = 0
    tp = fp = fn = 0
    for row in rows:
        pred = pipe.predict(
            current_caption=row.get("current_caption", ""),
            true_image_context=row.get("true_image_context", ""),
            vdt_label=row.get("vdt_label"),
            vdt_score=row.get("vdt_score"),
            sample_id=row.get("sample_id", ""),
            image_id=row.get("image_id", ""),
            domain=row.get("domain", "demo"),
        )
        gold_type = row.get("gold_mismatch_type")
        pred_type = pred.get("mismatch_type")
        gold_fields = norm_fields(row.get("gold_conflict_fields"))
        pred_fields = norm_fields(pred.get("conflict_fields"))
        type_ok += int(gold_type == pred_type)
        exact += int(gold_fields == pred_fields)
        for f in FIELDS:
            g, pr = f in gold_fields, f in pred_fields
            if g and pr:
                tp += 1
            elif pr and not g:
                fp += 1
            elif g and not pr:
                fn += 1
        outputs.append({"input": row, "prediction": pred, "checks": {"type_ok": gold_type == pred_type, "fields_exact": gold_fields == pred_fields}})
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "dataset": str(input_path.relative_to(ROOT)),
        "n": n,
        "purpose": "curated system-demo smoke set; not the final large-scale attribution experiment",
        "mismatch_type_accuracy": type_ok / n if n else 0.0,
        "conflict_field_precision": precision,
        "conflict_field_recall": recall,
        "conflict_field_micro_f1": f1,
        "exact_match_rate": exact / n if n else 0.0,
    }
    payload = {"summary": summary, "outputs": outputs}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_path}")
    # Demo set should pass exactly; if it fails, the UI examples are inconsistent.
    return 0 if summary["mismatch_type_accuracy"] >= 0.75 and summary["conflict_field_micro_f1"] >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
