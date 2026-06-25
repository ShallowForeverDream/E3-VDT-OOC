from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect VDT-COVE-Attr v2 experiment tables into Markdown.")
    ap.add_argument("--output", default="outputs/report_tables_v2.md")
    args = ap.parse_args()
    rows = []
    rows.append("# VDT-COVE-Attr v2 Experiment Tables\n")
    rows.append("## Table 1. VDT strict baseline reproduction\n")
    rows.append("| Method | Target domain | Batch size | F1 | Acc | AUC | Status |\n|---|---|---:|---:|---:|---:|---|\n")
    rows.append("| VDT strict BLIP-2/GaussianBlur | bbc,guardian | 128 | 0.7353 | 0.7383 | 0.7398 | completed |\n")
    rows.append("| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 128 | - | - | - | failed: CUDA OOM |\n")
    rows.append("| VDT strict BLIP-2/GaussianBlur | usa_today,washington_post | 64 | 0.8032 | 0.8032 | 0.8028 | completed |\n\n")

    ctx = read_json("outputs/cove_lite_context_pairs.jsonl.stats.json")
    rows.append("## Table 2. COVE-lite true-context construction\n")
    rows.append("| Total | Kept | Missing IDs | Missing text | Missing true context | Coverage |\n|---:|---:|---:|---:|---:|---:|\n")
    if ctx:
        st = ctx.get("stats", ctx)
        total = st.get("total", 0)
        kept = st.get("kept", 0)
        cov = kept / total if total else 0.0
        rows.append(f"| {total} | {kept} | {st.get('missing_ids', 0)} | {st.get('missing_text', 0)} | {st.get('missing_true_context', 0)} | {cov:.4f} |\n\n")
    else:
        rows.append("| 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |\n\n")

    ev = read_json("outputs/event_tuples_v2.jsonl.stats.json")
    rows.append("## Table 3. Event tuple extraction coverage\n")
    rows.append("| Extractor | Records | Missing text/context | current_entities | current_locations | current_times | current_event_types | current_relations |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    if ev:
        fc = ev.get("field_presence_counts", {})
        rows.append(f"| {ev.get('extractor', 'enhanced')} | {ev.get('records', 0)} | {ev.get('missing_text_or_context', 0)} | {fc.get('current_entities', 0)} | {fc.get('current_locations', 0)} | {fc.get('current_times', 0)} | {fc.get('current_event_types', 0)} | {fc.get('current_relations', 0)} |\n\n")
    else:
        rows.append("| rule/enhanced/NLI | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |\n\n")

    nli = read_json("outputs/field_nli_attribution_v2.jsonl.stats.json")
    rows.append("## Table 4. V2 attribution distribution\n")
    rows.append("| Records | NLI backend | Main mismatch distribution | Conflict field distribution |\n|---:|---|---|---|\n")
    if nli:
        rows.append(f"| {nli.get('records', 0)} | {nli.get('nli_backend_counts', {})} | {nli.get('v2_mismatch_type_counts', {})} | {nli.get('v2_conflict_field_counts', {})} |\n\n")
    else:
        rows.append("| 待跑 | 待跑 | 待跑 | 待跑 |\n\n")

    metrics = read_json("outputs/attribution_eval_v2_metrics.json")
    rows.append("## Table 5. Attribution evaluation on manual gold set\n")
    rows.append("| Method | Matched | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |\n|---|---:|---:|---:|---:|---:|\n")
    if metrics:
        for name, res in metrics.items():
            rows.append(f"| {name} | {res.get('matched', 0)} | {res.get('mismatch_type_accuracy', 0):.4f} | {res.get('conflict_field_micro_f1', 0):.4f} | {res.get('conflict_field_macro_f1', 0):.4f} | {res.get('exact_match_rate', 0):.4f} |\n")
        rows.append("\n")
    else:
        rows.append("| majority/random/rule/NLI | 待人工标注 | 待测 | 待测 | 待测 | 待测 |\n\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
