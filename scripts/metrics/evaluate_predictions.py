from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from e3vdt.metrics.classification import classification_report

def main():
    ap=argparse.ArgumentParser(description="Evaluate JSONL predictions."); ap.add_argument("--input", required=True); ap.add_argument("--gold-field", default="label"); ap.add_argument("--pred-field", default="pred"); ap.add_argument("--output", default=None); args=ap.parse_args()
    y_true=[]; y_pred=[]
    with open(args.input,encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec=json.loads(line); y_true.append(str(rec[args.gold_field])); y_pred.append(str(rec[args.pred_field]))
    report=classification_report(y_true,y_pred); text=json.dumps(report, ensure_ascii=False, indent=2); print(text)
    if args.output: Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(text,encoding="utf-8")
if __name__ == "__main__": main()
