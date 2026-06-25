from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

TYPE_EN = {
    "正常匹配/无错配": "benign illustrative image",
    "主体错配": "entity mismatch",
    "地点错配": "location mismatch",
    "时间错配": "temporal mismatch",
    "事件类型错配": "event-type mismatch",
    "行为关系错配": "relation mismatch",
    "上下文缺失": "context omission",
    "证据不足/无法判断": "uncertain / evidence insufficient",
}
FIELD_EN = {
    "主体": "entity",
    "地点": "location",
    "时间": "time",
    "事件类型": "event_type",
    "行为关系": "relation",
    "上下文缺失": "context_omission",
    "证据不足": "evidence_insufficient",
}


def parse_label(x: str) -> int | None:
    if "OOC" in x and "Non" not in x:
        return 1
    if "Non-OOC" in x or "正常" in x:
        return 0
    return None


def split_fields(x: str) -> List[str]:
    if not x:
        return []
    parts = [p.strip() for p in x.replace(",", "；").replace(";", "；").split("；") if p.strip()]
    return [FIELD_EN.get(p, p) for p in parts]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Import edited Chinese annotation CSV files back to JSONL.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", default="examples/attribution_eval_set.jsonl")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for inp in args.inputs:
        for r in read_csv(Path(inp)):
            gold_cn = (r.get("人工最终错配类型") or "").strip()
            field_cn = (r.get("人工最终冲突字段") or "").strip()
            status = (r.get("标注状态") or "").strip()
            rows.append({
                "sample_id": r.get("样本ID", ""),
                "image_id": r.get("image_id", ""),
                "text_id": r.get("text_id", ""),
                "split": r.get("split", ""),
                "domain": r.get("domain", ""),
                "label": parse_label(r.get("数据标签", "")),
                "current_caption": r.get("当前文本", ""),
                "true_image_context": r.get("图像真实上下文", ""),
                "weak_mismatch_type": TYPE_EN.get((r.get("系统建议错配类型") or "").strip(), (r.get("系统建议错配类型") or "").strip()),
                "weak_conflict_fields": split_fields(r.get("系统建议冲突字段", "")),
                "gold_mismatch_type": TYPE_EN.get(gold_cn, gold_cn),
                "gold_conflict_fields": split_fields(field_cn),
                "annotator": r.get("标注人", ""),
                "rationale": r.get("人工理由", ""),
                "annotation_status": "done" if status.lower() == "done" or status == "已完成" else status or "review_needed",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "records": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
