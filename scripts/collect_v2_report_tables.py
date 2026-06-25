from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv_rows(path: str):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    rows.append("| Total | Available before cap | Sampled/kept | Missing IDs | Missing text | Missing true context | Coverage before cap |\n|---:|---:|---:|---:|---:|---:|---:|\n")
    if ctx:
        st = ctx.get("stats", ctx)
        total = st.get("total", 0)
        kept = st.get("kept", 0)
        available = st.get("available_before_cap", kept)
        cov = available / total if total else 0.0
        rows.append(f"| {total} | {available} | {kept} | {st.get('missing_ids', 0)} | {st.get('missing_text', 0)} | {st.get('missing_true_context', 0)} | {cov:.4f} |\n\n")
    else:
        rows.append("| 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |\n\n")

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

    cf = read_json("outputs/counterfactual/counterfactual_generation_stats.json")
    rows.append("## Table 6. Controlled counterfactual attribution data\n")
    rows.append("| Edit type | Generated/kept | Attempts | Keep rate |\n|---|---:|---:|---:|\n")
    if cf:
        kept = cf.get("kept", {})
        attempted = cf.get("attempted", {})
        rates = cf.get("keep_rate", {})
        rows.append(f"| none / positive | {kept.get('none', 0)} | {kept.get('none', 0)} | 1.0000 |\n")
        for edit in ["location_swap", "time_swap", "entity_swap"]:
            rows.append(f"| {edit} | {kept.get(edit, 0)} | {attempted.get(edit, 0)} | {float(rates.get(edit, 0.0)):.4f} |\n")
        rows.append("\n")
    else:
        rows.append("| none/location/time/entity | 待跑 | 待跑 | 待跑 |\n\n")

    head = read_json("outputs/counterfactual/attribution_head_metrics.json")
    rows.append("## Table 7. Attribution head on controlled counterfactual test set\n")
    rows.append("| Method | N | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |\n|---|---:|---:|---:|---:|---:|\n")
    if head:
        for name, res in head.get("results", {}).items():
            rows.append(f"| {name} | {res.get('n', 0)} | {res.get('mismatch_type_accuracy', 0):.4f} | {res.get('conflict_field_micro_f1', 0):.4f} | {res.get('conflict_field_macro_f1', 0):.4f} | {res.get('exact_match_rate', 0):.4f} |\n")
        rows.append("\n")
    else:
        rows.append("| majority/rule/NLI/attr_head | 待跑 | 待测 | 待测 | 待测 | 待测 |\n\n")

    leak = read_json("outputs/counterfactual/leakage_check.json")
    rows.append("## Table 8. Counterfactual split leakage audit\n")
    rows.append("| Check | Value |\n|---|---:|\n")
    if leak:
        rows.append(f"| source_sample_id leakage | {leak.get('source_sample_id_leakage', 0)} |\n")
        rows.append(f"| image_id leakage | {leak.get('image_id_leakage', 0)} |\n")
        rows.append(f"| text_id leakage | {leak.get('text_id_leakage', 0)} |\n")
        rows.append(f"| cross-split duplicate edited caption | {leak.get('cross_split_duplicate_edited_caption', 0)} |\n\n")
    else:
        rows.append("| leakage check | 待跑 |\n\n")

    scaling = read_csv_rows("outputs/counterfactual_scaling_results.csv")
    rows.append("## Table 9. Controlled counterfactual scaling curve\n")
    rows.append("| MaxPerType | Method | Train | Test | Type Acc | Field Micro-F1 | Exact Match | Leakage | Transformers |\n|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
    if scaling:
        for r in scaling:
            rows.append(f"| {r.get('max_per_type', '')} | {r.get('method', '')} | {r.get('train_rows', '')} | {r.get('test_rows', '')} | {float(r.get('type_acc') or 0):.4f} | {float(r.get('field_micro_f1') or 0):.4f} | {float(r.get('exact_match') or 0):.4f} | {r.get('source_sample_id_leakage', '')}/{r.get('image_id_leakage', '')} | {r.get('used_transformers', '')} |\n")
        rows.append("\n")
    else:
        rows.append("| 80/200/1000/3000 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |\n\n")

    real = read_json("outputs/real_ooc_attribution_eval_metrics.json")
    rows.append("## Table 10. Manual real OOC attribution evaluation\n")
    rows.append("| Method | Matched | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |\n|---|---:|---:|---:|---:|---:|\n")
    if real and real.get("gold_records_done", 0):
        for name, res in real.get("methods", {}).items():
            rows.append(f"| {name} | {res.get('matched', 0)} | {res.get('mismatch_type_accuracy', 0):.4f} | {res.get('conflict_field_micro_f1', 0):.4f} | {res.get('conflict_field_macro_f1', 0):.4f} | {res.get('exact_match_rate', 0):.4f} |\n")
        rows.append("\n")
    else:
        rows.append("| rule/NLI/counterfactual-trained attr head | 待人工标注 | 待测 | 待测 | 待测 | 待测 |\n\n")

    ntc = read_json("outputs/no_true_context_attr/no_true_context_attr_metrics.json")
    rows.append("## Table 11. No-true-context image+caption attribution head\n")
    rows.append("| Method | Uses true context at inference? | N | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |\n|---|---|---:|---:|---:|---:|---:|\n")
    if ntc:
        for name, res in ntc.get("results", {}).items():
            rows.append(f"| {name} | {ntc.get('uses_true_context_at_inference', False)} | {res.get('n', 0)} | {res.get('mismatch_type_accuracy', 0):.4f} | {res.get('conflict_field_micro_f1', 0):.4f} | {res.get('conflict_field_macro_f1', 0):.4f} | {res.get('exact_match_rate', 0):.4f} |\n")
        rows.append("\n")
    else:
        rows.append("| image-caption attr head | False | 待跑 | 待测 | 待测 | 待测 | 待测 |\n\n")

    ntc_scaling = read_csv_rows("outputs/no_true_context_scaling_results.csv")
    rows.append("## Table 12. No-true-context attribution scaling curve\n")
    rows.append("| MaxPerType | Method | Train | Test | Type Acc | Field Micro-F1 | Exact Match | Counts none/location/time/entity/different | Leakage |\n|---:|---|---:|---:|---:|---:|---:|---|---|\n")
    if ntc_scaling:
        for r in ntc_scaling:
            counts = f"{r.get('none_count','')}/{r.get('location_swap_count','')}/{r.get('time_swap_count','')}/{r.get('entity_swap_count','')}/{r.get('different_event_count','0')}"
            leakage = f"{r.get('source_sample_id_leakage','')}/{r.get('image_id_leakage','')}/{r.get('text_id_leakage','')}/{r.get('cross_split_duplicate_edited_caption','')}"
            rows.append(f"| {r.get('max_per_type', '')} | {r.get('method', '')} | {r.get('train_rows', '')} | {r.get('test_rows', '')} | {float(r.get('type_acc') or 0):.4f} | {float(r.get('field_micro_f1') or 0):.4f} | {float(r.get('exact_match') or 0):.4f} | {counts} | {leakage} |\n")
        rows.append("\n")
    else:
        rows.append("| 80/200/1000 | 待跑 | 待跑 | 待跑 | 待测 | 待测 | 待测 | 待统计 | 待查 |\n\n")

    ntc5 = read_json("outputs/no_true_context_attr_5way_1000/no_true_context_attr_metrics.json")
    ntc5_stats = read_json("outputs/no_true_context_attr_5way_1000/counterfactual_generation_stats.json")
    rows.append("## Table 13. No-true-context five-class attribution with filtered original OOC\n")
    rows.append("| Method | Selected? | N | Type Acc | Field Micro-F1 | Exact Match | Counts none/location/time/entity/different |\n|---|---|---:|---:|---:|---:|---|\n")
    if ntc5:
        counts_obj = ntc5_stats.get("type_counts", {}) if ntc5_stats else {}
        counts = f"{counts_obj.get('none',0)}/{counts_obj.get('location_swap',0)}/{counts_obj.get('time_swap',0)}/{counts_obj.get('entity_swap',0)}/{counts_obj.get('different_event_original_ooc',0)}"
        selected = ntc5.get("selected_model_name", "")
        for name, res in ntc5.get("results", {}).items():
            rows.append(f"| {name} | {str(name == selected)} | {res.get('n', 0)} | {res.get('mismatch_type_accuracy', 0):.4f} | {res.get('conflict_field_micro_f1', 0):.4f} | {res.get('exact_match_rate', 0):.4f} | {counts} |\n")
        rows.append("\n")
    else:
        rows.append("| five-class LR/MLP | 待跑 | 待跑 | 待测 | 待测 | 待测 | 待统计 |\n\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
