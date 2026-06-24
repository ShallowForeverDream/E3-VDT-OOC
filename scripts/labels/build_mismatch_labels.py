from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from e3vdt.inference.pipeline import E3VDTPipeline

def main():
    ap=argparse.ArgumentParser(description="Build weak mismatch-type labels from JSONL records.")
    ap.add_argument("--input", required=True, help="JSONL with fields: id, text, image_context")
    ap.add_argument("--output", required=True)
    args=ap.parse_args(); pipe=E3VDTPipeline(); out_path=Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True); n=0
    with open(args.input,encoding="utf-8") as f, open(out_path,"w",encoding="utf-8") as out:
        for line in f:
            if not line.strip(): continue
            rec=json.loads(line); pred=pipe.predict_dict(text=rec.get("text",""), image_context=rec.get("image_context",""))
            rec.update({"weak_label":pred["label"],"weak_mismatch_type":pred["mismatch_type"],"weak_conflict_fields":pred["conflict_fields"],"event_scores":pred["event_scores"]})
            out.write(json.dumps(rec, ensure_ascii=False)+"\n"); n+=1
    print(f"wrote {n} records to {out_path}")
if __name__ == "__main__": main()
