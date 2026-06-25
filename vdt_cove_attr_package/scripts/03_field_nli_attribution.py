from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

PRIORITY=["location","time","entity","event_type","relation"]
TYPE_BY_FIELD={"entity":"entity mismatch","location":"location mismatch","time":"temporal mismatch","event_type":"event-type mismatch","relation":"relation mismatch"}
FIELD_TO_KEY={"entity":"entities","location":"locations","time":"times","event_type":"event_types","relation":"relations"}


def make_hypothesis(field: str, value: str) -> str:
    if field == "location": return f"The image event happened in {value}."
    if field == "time": return f"The image event happened at {value}."
    if field == "entity": return f"{value} was involved in the image event."
    if field == "event_type": return f"The image event is about {value}."
    if field == "relation": return f"The image event involves the action {value}."
    return f"The image event is related to {value}."


def simple_overlap(a: List[str], b: List[str]) -> float:
    aa={str(x).lower().strip() for x in a if str(x).strip()}; bb={str(x).lower().strip() for x in b if str(x).strip()}
    if not aa and not bb: return 0.5
    if not aa or not bb: return 0.0
    inter=len(aa & bb); union=len(aa | bb)
    return inter/union if union else 0.0


def load_nli(model_name: str, device: int):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    if device >= 0 and torch.cuda.is_available():
        model=model.to(f"cuda:{device}")
    model.eval()
    id2label={int(k):v.lower() for k,v in model.config.id2label.items()}
    return tok, model, id2label


def nli_probs(tok, model, id2label, premise: str, hypothesis: str, device: int) -> Dict[str,float]:
    import torch
    enc=tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    if device >= 0 and torch.cuda.is_available():
        enc={k:v.to(f"cuda:{device}") for k,v in enc.items()}
    with torch.no_grad():
        logits=model(**enc).logits[0]
        probs=torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
    out={"entailment":0.0,"neutral":0.0,"contradiction":0.0}
    for i,p in enumerate(probs):
        lab=id2label.get(i,str(i)).lower()
        if "entail" in lab: out["entailment"]=float(p)
        elif "contr" in lab: out["contradiction"]=float(p)
        elif "neutral" in lab: out["neutral"]=float(p)
    if sum(out.values()) == 0:  # fallback for label_0 style models
        out["contradiction"], out["neutral"], out["entailment"] = probs[:3]
    return out


def score_field(field: str, current_vals: List[str], true_vals: List[str], true_context: str, tok, model, id2label, args) -> Dict[str,Any]:
    overlap=simple_overlap(current_vals, true_vals)
    if not true_context.strip() or not current_vals:
        return {"label":"evidence_insufficient", "overlap":overlap, "contradiction":0.0, "entailment":0.0, "neutral":0.0, "details":[]}
    details=[]; best={"contradiction":0.0,"entailment":0.0,"neutral":0.0}
    for v in current_vals[:args.max_values_per_field]:
        hyp=make_hypothesis(field, str(v))
        if tok is not None:
            probs=nli_probs(tok,model,id2label,true_context,hyp,args.device)
        else:
            probs={"contradiction":1.0-overlap,"entailment":overlap,"neutral":0.0}
        details.append({"value":v,"hypothesis":hyp,**probs})
        for k in best:
            best[k]=max(best[k],probs[k])
    if best["contradiction"] >= args.contradiction_threshold and best["contradiction"] >= best["entailment"]:
        label="contradiction"
    elif best["entailment"] >= args.entailment_threshold:
        label="entailment"
    else:
        label="neutral"
    return {"label":label, "overlap":overlap, **best, "details":details}


def main():
    ap=argparse.ArgumentParser(description="Field-wise NLI attribution for VDT-COVE-Attr.")
    ap.add_argument("--input", required=True, help="event_tuples.jsonl from 02_extract_events.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--nli-model", default="facebook/bart-large-mnli")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-nli", action="store_true", help="Use overlap fallback, for debugging only.")
    ap.add_argument("--contradiction-threshold", type=float, default=0.55)
    ap.add_argument("--entailment-threshold", type=float, default=0.55)
    ap.add_argument("--max-values-per-field", type=int, default=3)
    args=ap.parse_args()
    tok=model=id2label=None
    if not args.no_nli:
        tok,model,id2label=load_nli(args.nli_model,args.device)
    out_path=Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    stats={"records":0,"method":"field_wise_nli" if not args.no_nli else "overlap_fallback","mismatch_type_counts":{}}
    with open(args.input,encoding="utf-8") as f, open(out_path,"w",encoding="utf-8") as out:
        for line in tqdm(f,desc="field NLI"):
            if not line.strip(): continue
            rec=json.loads(line)
            cur_evt=rec.get("current_event_tuple",{}) or {}; true_evt=rec.get("true_event_tuple",{}) or {}
            true_context=rec.get("true_image_context","") or ""
            field_nli={}; conflict_fields=[]; insufficient=[]
            for field,key in FIELD_TO_KEY.items():
                res=score_field(field, cur_evt.get(key,[]) or [], true_evt.get(key,[]) or [], true_context, tok, model, id2label, args)
                field_nli[field]=res
                if res["label"]=="contradiction": conflict_fields.append(field)
                if res["label"]=="evidence_insufficient": insufficient.append(field)
            if conflict_fields:
                main_field=next((x for x in PRIORITY if x in conflict_fields), conflict_fields[0])
                mtype=TYPE_BY_FIELD.get(main_field,"context omission")
            elif len(insufficient) >= 3:
                mtype="uncertain / evidence insufficient"; conflict_fields=["evidence_insufficient"]
            else:
                mtype="benign illustrative image"
            row=dict(rec)
            row.update({
                "pred_mismatch_type": mtype,
                "pred_conflict_fields": conflict_fields,
                "field_nli": field_nli,
                "attribution_method": stats["method"],
            })
            stats["records"] += 1
            stats["mismatch_type_counts"][mtype]=stats["mismatch_type_counts"].get(mtype,0)+1
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    Path(str(out_path)+".stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(stats,indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()
