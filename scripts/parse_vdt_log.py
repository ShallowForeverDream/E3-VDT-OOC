from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

FIELD_PATTERNS = {
    "accuracy_at_eer": r"^Accuracy at EER:\s*([-+0-9.eE]+)",
    "eer": r"^EER:\s*([-+0-9.eE]+)",
    "threshold_at_eer": r"^Threshold at EER:\s*([-+0-9.eE]+)",
    "auc": r"^AUC score:\s*([-+0-9.eE]+)",
    "f1": r"^f1:\s*([-+0-9.eE]+)",
    "acc": r"^Acc:\s*([-+0-9.eE]+)",
    "f1_real": r"^f1_real:\s*([-+0-9.eE]+)",
    "f1_fake": r"^f1_fake:\s*([-+0-9.eE]+)",
}
CM_RE = re.compile(r"\[\[(\d+)\s+(\d+)\]\s*\n\s*\[\s*(\d+)\s+(\d+)\]\]")


def parse_blocks(text: str) -> list[dict]:
    starts = [m.start() for m in re.finditer(r"Accuracy at EER:", text)]
    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end]
        item = {}
        for key, pattern in FIELD_PATTERNS.items():
            m = re.search(pattern, chunk, re.MULTILINE)
            if m:
                item[key] = float(m.group(1))
        cm = CM_RE.search(chunk)
        if cm:
            item["confusion_matrix"] = [[int(cm.group(1)), int(cm.group(2))], [int(cm.group(3)), int(cm.group(4))]]
        if item:
            blocks.append(item)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse VDT train_stdout.log metrics blocks.")
    parser.add_argument("log", type=Path, help="Path to train_stdout.log")
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    blocks = parse_blocks(text)
    best = max(blocks, key=lambda x: x.get("f1", -1.0), default=None)
    result = {"log": str(args.log), "num_blocks": len(blocks), "best_by_f1": best, "blocks": blocks}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if blocks else 1


if __name__ == "__main__":
    raise SystemExit(main())
