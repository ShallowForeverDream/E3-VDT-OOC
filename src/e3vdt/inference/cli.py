from __future__ import annotations
import argparse, json
from pathlib import Path
from e3vdt.inference.pipeline import E3VDTPipeline

def main() -> None:
    ap=argparse.ArgumentParser(description="Run E3-VDT-OOC demo inference.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-context", default="")
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    result=E3VDTPipeline().predict_dict(text=args.text, image_path=args.image, image_context=args.image_context)
    text=json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
if __name__ == "__main__": main()
