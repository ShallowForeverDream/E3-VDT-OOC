from __future__ import annotations
import argparse, json, random, csv
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import Counter, defaultdict

FIELDS=["entity","location","time","event_type","relation","context_omission","evidence_insufficient"]
TYPES=["entity mismatch","location mismatch","temporal mismatch","event-type mismatch","relation mismatch","context omission","uncertain / evidence insufficient","benign illustrative image"]


def load_jsonl(path: str) -> List[Dict[str,Any]]:
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def norm_type(x):
    return str(x or "").strip().lower()


def norm_fields(xs) -> Set[str]:
    if isinstance(xs, str): xs=[xs]
    return {str(x).strip().lower() for x in (xs or []) if str(x).strip()}


def evaluate(pred_rows: List[Dict[str,Any]], gold_rows: List[Dict[str,Any]], method_name: str) -> Dict[str,Any]:
    pred_by_id={str(r.get("sample_id")): r for r in pred_rows}
    type_correct=0; n=0; exact=0
    tp=Counter(); fp=Counter(); fn=Counter()
    for g in gold_rows:
        if str(g.get("annotation_status","done")).lower() not in {"done","gold",""}:
            continue
        sid=str(g.get("sample_id"))
        p=pred_by_id.get(sid, {})
        gt_type=norm_type(g.get("gold_mismatch_type"))
        pred_type=norm_type(p.get("pred_mismatch_type") or p.get("weak_mismatch_type"))
        gt_fields=norm_fields(g.get("gold_conflict_fields"))
        pred_fields=norm_fields(p.get("pred_conflict_fields") or p.get("weak_conflict_fields"))
        if not gt_type and not gt_fields: continue
        n += 1
        if pred_type == gt_type: type_correct += 1
        if pred_fields == gt_fields: exact += 1
        for f in FIELDS:
            if f in pred_fields and f in gt_fields: tp[f]+=1
            elif f in pred_fields and f not in gt_fields: fp[f]+=1
            elif f not in pred_fields and f in gt_fields: fn[f]+=1
    micro_tp=sum(tp.values()); micro_fp=sum(fp.values()); micro_fn=sum(fn.values())
    micro_p=micro_tp/(micro_tp+micro_fp) if micro_tp+micro_fp else 0.0
    micro_r=micro_tp/(micro_tp+micro_fn) if micro_tp+micro_fn else 0.0
    micro_f1=2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0.0
    macro=[]
    per_field={}
    for f in FIELDS:
        p=tp[f]/(tp[f]+fp[f]) if tp[f]+fp[f] else 0.0
        r=tp[f]/(tp[f]+fn[f]) if tp[f]+fn[f] else 0.0
        f1=2*p*r/(p+r) if p+r else 0.0
        per_field[f]={"precision":p,"recall":r,"f1":f1,"tp":tp[f],"fp":fp[f],"fn":fn[f]}
        macro.append(f1)
    return {
        "method": method_name,
        "n": n,
        "mismatch_type_accuracy": type_correct/n if n else 0.0,
        "conflict_field_micro_precision": micro_p,
        "conflict_field_micro_recall": micro_r,
        "conflict_field_micro_f1": micro_f1,
        "conflict_field_macro_f1": sum(macro)/len(macro) if macro else 0.0,
        "exact_match_rate": exact/n if n else 0.0,
        "per_field": per_field,
    }


def make_majority(gold_rows):
    types=[norm_type(r.get("gold_mismatch_type")) for r in gold_rows if r.get("gold_mismatch_type")]
    fields=[]
    for r in gold_rows: fields.extend(list(norm_fields(r.get("gold_conflict_fields"))))
    maj_type=Counter(types).most_common(1)[0][0] if types else "uncertain / evidence insufficient"
    maj_field=Counter(fields).most_common(1)[0][0] if fields else "evidence_insufficient"
    return [{"sample_id":r.get("sample_id"),"pred_mismatch_type":maj_type,"pred_conflict_fields":[maj_field]} for r in gold_rows]


def make_random(gold_rows, seed=42):
    random.seed(seed)
    return [{"sample_id":r.get("sample_id"),"pred_mismatch_type":random.choice(TYPES),"pred_conflict_fields":[random.choice(FIELDS)]} for r in gold_rows]


def main():
    ap=argparse.ArgumentParser(description="Evaluate attribution predictions against gold labels.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    gold=load_jsonl(args.gold); pred=load_jsonl(args.pred)
    results=[]
    results.append(evaluate(make_majority(gold), gold, "majority_baseline"))
    results.append(evaluate(make_random(gold), gold, "random_baseline"))
    results.append(evaluate(pred, gold, "vdt_cove_attr_prediction"))
    out={"gold":args.gold,"pred":args.pred,"results":results}
    out_path=Path(args.output); out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    csv_path=out_path.with_suffix(".csv")
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f, fieldnames=["method","n","mismatch_type_accuracy","conflict_field_micro_f1","conflict_field_macro_f1","exact_match_rate"])
        wr.writeheader()
        for r in results:
            wr.writerow({k:r.get(k) for k in wr.fieldnames})
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()
