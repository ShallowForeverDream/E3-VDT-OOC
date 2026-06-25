from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e3vdt.attribution.evidence_relevance_v2 import score_evidence_relevance
from e3vdt.attribution.field_nli_v2 import NLIPredictor, run_field_nli
from e3vdt.attribution.graph_alignment_v2 import align_event_graphs


FIELD_TO_TYPE = {
    "entity": "entity mismatch",
    "location": "location mismatch",
    "time": "temporal mismatch",
    "event_type": "event-type mismatch",
    "relation": "relation mismatch",
}


def is_non_ooc(rec: Dict[str, Any]) -> bool:
    label = rec.get("label")
    if label == 0:
        return True
    if isinstance(label, str) and label.strip().lower() in {"0", "false", "non-ooc", "non_ooc", "real", "match", "matched"}:
        return True
    weak_label = str(rec.get("weak_label", "")).strip().lower()
    return weak_label in {"non-ooc", "non_ooc", "real", "match", "matched"}


def is_different_event(conflicts: List[str]) -> bool:
    """Return true when conflict pattern is better described as whole-event mismatch.

    Real OOC samples are often not a single atomic slot swap.  If several core
    event fields contradict at once, especially event_type plus another field,
    the primary type should be "different-event mismatch" while conflict_fields
    still preserves the detailed multi-label evidence.
    """

    core = {c for c in conflicts if c in {"entity", "location", "time", "event_type", "relation"}}
    return len(core) >= 3 or ("event_type" in core and len(core) >= 2)


def final_decision(rec: Dict[str, Any], field: Dict[str, Any], evidence: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    if is_non_ooc(rec):
        return {
            "mismatch_type": "benign illustrative image",
            "conflict_fields": [],
            "attribution_reason": "non_ooc_no_attribution",
        }
    if evidence.get("evidence_sufficiency") != "sufficient":
        return {
            "mismatch_type": "uncertain / evidence insufficient",
            "conflict_fields": ["evidence_insufficient"],
            "attribution_reason": evidence.get("evidence_reason", "insufficient"),
        }
    conflicts = list(field.get("conflict_fields", []))
    for g in graph.get("graph_conflicts", []):
        if g not in conflicts:
            conflicts.append(g)
    if conflicts:
        if is_different_event(conflicts):
            return {
                "mismatch_type": "different-event mismatch",
                "conflict_fields": conflicts,
                "attribution_reason": "multi_field_event_contradiction",
            }
        mt = field.get("mismatch_type") or "context omission"
        if mt == "context omission" and conflicts:
            mt = FIELD_TO_TYPE.get(conflicts[0], "context omission")
        return {"mismatch_type": mt, "conflict_fields": conflicts, "attribution_reason": "field_or_graph_contradiction"}
    return {"mismatch_type": "benign illustrative image", "conflict_fields": [], "attribution_reason": "no_field_contradiction"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VDT-COVE-Attr v2 field-wise NLI attribution.")
    ap.add_argument("--input", required=True, help="JSONL produced by extract_event_tuples_v2.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="facebook/bart-large-mnli")
    ap.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 first CUDA device")
    ap.add_argument("--no-transformers", action="store_true", help="Use deterministic lexical fallback only")
    ap.add_argument("--contradiction-threshold", type=float, default=0.60)
    ap.add_argument("--entailment-threshold", type=float, default=0.60)
    args = ap.parse_args()

    nli = NLIPredictor(model_name=args.model, device=args.device, use_transformers=not args.no_transformers)
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    type_counts = Counter()
    field_counts = Counter()
    backend_counts = Counter()
    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec: Dict[str, Any] = json.loads(line)
            current_caption = rec.get("current_caption") or rec.get("text") or ""
            true_context = rec.get("true_image_context") or rec.get("image_context") or ""
            cur_event = rec.get("current_event", {})
            true_event = rec.get("true_event", {})
            evidence = score_evidence_relevance(current_caption, true_context, cur_event, true_event)
            field = run_field_nli(cur_event, true_event, true_context, nli, contradiction_threshold=args.contradiction_threshold, entailment_threshold=args.entailment_threshold)
            graph = align_event_graphs(cur_event, true_event)
            decision = final_decision(rec, field, evidence, graph)
            row = dict(rec)
            row.update({
                "v2_mismatch_type": decision["mismatch_type"],
                "v2_conflict_fields": decision["conflict_fields"],
                "v2_attribution_reason": decision["attribution_reason"],
                "field_nli": field.get("field_nli", {}),
                "evidence_relevance": evidence,
                "graph_alignment": graph,
                "v2_method": "cove_lite_field_nli_evidence_graph_v2",
            })
            type_counts[row["v2_mismatch_type"]] += 1
            for cf in row["v2_conflict_fields"]:
                field_counts[cf] += 1
            for fv in row["field_nli"].values():
                backend = fv.get("scores", {}).get("_backend") if isinstance(fv, dict) else None
                if backend:
                    backend_counts[str(backend)] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    stats = {"input": str(in_path), "output": str(out_path), "records": n, "v2_mismatch_type_counts": dict(type_counts), "v2_conflict_field_counts": dict(field_counts), "nli_backend_counts": dict(backend_counts), "nli_model": args.model, "used_transformers": not args.no_transformers}
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
