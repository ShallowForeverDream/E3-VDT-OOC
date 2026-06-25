from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str,Any]]:
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def main():
    ap=argparse.ArgumentParser(description="Build human attribution annotation candidates.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-only", action="store_true")
    args=ap.parse_args()
    rows=load_jsonl(args.input)
    if args.hard_only:
        # select likely hard cases: high label=1 and no trivial evidence insufficiency
        rows=[r for r in rows if r.get("label") in {1,"1",True} and r.get("pred_mismatch_type") != "uncertain / evidence insufficient"]
    random.seed(args.seed)
    random.shuffle(rows)
    rows=rows[:args.n]
    out_path=Path(args.output); out_path.parent.mkdir(parents=True,exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as out:
        for r in rows:
            item={
                "sample_id": r.get("sample_id"),
                "image_id": r.get("image_id"),
                "label": r.get("label"),
                "current_caption": r.get("current_caption"),
                "true_image_context": r.get("true_image_context"),
                "pred_mismatch_type": r.get("pred_mismatch_type"),
                "pred_conflict_fields": r.get("pred_conflict_fields", []),
                "field_nli_summary": {k:v.get("label") for k,v in (r.get("field_nli") or {}).items()},
                "gold_mismatch_type": "",
                "gold_conflict_fields": [],
                "rationale": "",
                "annotator": "",
                "annotation_status": "todo",
            }
            out.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(out_path),"records":len(rows)},indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()
