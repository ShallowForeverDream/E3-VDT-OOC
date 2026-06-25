from __future__ import annotations
import argparse, json
from pathlib import Path


def read_json(path):
    p=Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--out", default="outputs/report_tables.md")
    args=ap.parse_args()
    od=Path(args.output_dir)
    context=read_json(od/"cove_lite_context_pairs.jsonl.stats.json")
    events=read_json(od/"event_tuples.jsonl.stats.json")
    pred=read_json(od/"field_nli_attribution.jsonl.stats.json")
    evalr=read_json(od/"attribution_eval_metrics.json")
    lines=["# Report Tables",""]
    lines.append("## COVE-lite Context Coverage")
    if context:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k in ["total","kept","coverage","missing_ids","missing_text","missing_true_context","metadata_index_size"]:
            if k in context: lines.append(f"| {k} | {context[k]} |")
    lines.append("\n## Event Extraction Stats")
    if events:
        lines.append("```json\n"+json.dumps(events,indent=2,ensure_ascii=False)+"\n```")
    lines.append("\n## Attribution Prediction Stats")
    if pred:
        lines.append("```json\n"+json.dumps(pred,indent=2,ensure_ascii=False)+"\n```")
    lines.append("\n## Attribution Evaluation")
    if evalr:
        lines.append("| Method | N | Type Acc | Field Micro-F1 | Field Macro-F1 | Exact Match |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in evalr.get("results",[]):
            lines.append(f"| {r['method']} | {r['n']} | {r['mismatch_type_accuracy']:.4f} | {r['conflict_field_micro_f1']:.4f} | {r['conflict_field_macro_f1']:.4f} | {r['exact_match_rate']:.4f} |")
    out_path=Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text("\n".join(lines),encoding="utf-8")
    print("saved", out_path)

if __name__ == "__main__":
    main()
