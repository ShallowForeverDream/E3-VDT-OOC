import json
from pathlib import Path

from e3vdt.inference.pipeline import E3VDTPipeline


ROOT = Path(__file__).resolve().parents[1]


def test_demo_cases_match_documented_expected_outputs():
    pipe = E3VDTPipeline()
    cases = [
        json.loads(line)
        for line in (ROOT / "examples" / "demo_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert cases, "demo_cases.jsonl should contain answerable presentation examples"

    for case in cases:
        out = pipe.predict_dict(text=case["text"], image_context=case.get("image_context", ""))
        assert out["label"] == case["expected_label"], case["id"]
        assert out["mismatch_type"] == case["expected_mismatch_type"], case["id"]
        assert out["conflict_fields"] == case["expected_conflict_fields"], case["id"]


def test_reproduction_metrics_json_has_report_fields():
    rows = json.loads((ROOT / "examples" / "reproduction_metrics.json").read_text(encoding="utf-8"))
    assert rows
    for row in rows:
        assert row["id"]
        assert row["status"]
        assert row["target_domain_arg"]
        if row.get("metrics"):
            for key in ["f1", "acc", "auc"]:
                assert isinstance(row["metrics"][key], float)
