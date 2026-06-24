from __future__ import annotations
import argparse, json
from pathlib import Path

def is_hard_negative(rec, min_clip=0.28):
    if rec.get("weak_label") != "OOC": return False
    if float(rec.get("clip_similarity",0.0) or 0.0) >= min_clip: return True
    fields=set(rec.get("weak_conflict_fields") or [])
    return bool(fields & {"entity","location","time","event_type"})
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--min-clip", type=float, default=0.28); args=ap.parse_args()
    n=kept=0; Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input,encoding="utf-8") as f, open(args.output,"w",encoding="utf-8") as out:
        for line in f:
            if not line.strip(): continue
            n+=1; rec=json.loads(line)
            if is_hard_negative(rec,args.min_clip): out.write(json.dumps(rec, ensure_ascii=False)+"\n"); kept+=1
    print(f"kept {kept}/{n} hard negatives -> {args.output}")
if __name__ == "__main__": main()
