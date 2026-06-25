from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key_of(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("image_id") or row.get("current_caption") or "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Count key overlap between manual gold labels and current predictions.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    gold = {key_of(r) for r in load_jsonl(Path(args.gold)) if key_of(r)}
    pred = {key_of(r) for r in load_jsonl(Path(args.pred)) if key_of(r)}
    print(len(gold & pred))


if __name__ == "__main__":
    main()
