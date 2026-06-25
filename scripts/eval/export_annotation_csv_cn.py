from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

TYPE_CN = {
    "benign illustrative image": "正常匹配/无错配",
    "entity mismatch": "主体错配",
    "location mismatch": "地点错配",
    "temporal mismatch": "时间错配",
    "event-type mismatch": "事件类型错配",
    "relation mismatch": "行为关系错配",
    "context omission": "上下文缺失",
    "uncertain / evidence insufficient": "证据不足/无法判断",
}
FIELD_CN = {
    "entity": "主体",
    "location": "地点",
    "time": "时间",
    "event_type": "事件类型",
    "relation": "行为关系",
    "context_omission": "上下文缺失",
    "evidence_insufficient": "证据不足",
}

HEADERS = [
    "序号", "样本ID", "数据标签", "当前文本", "图像真实上下文",
    "系统建议错配类型", "系统建议冲突字段",
    "人工最终错配类型", "人工最终冲突字段", "人工理由", "标注人", "标注状态",
    "split", "domain", "image_id", "text_id",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cn_type(x: str) -> str:
    return TYPE_CN.get(x or "", x or "")


def cn_fields(xs: Any) -> str:
    if not xs:
        return ""
    if isinstance(xs, str):
        xs = [x.strip() for x in xs.replace(",", "；").split("；") if x.strip()]
    return "；".join(FIELD_CN.get(str(x), str(x)) for x in xs)


def label_cn(x: Any) -> str:
    if x == 1:
        return "OOC错配"
    if x == 0:
        return "Non-OOC正常"
    return str(x)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({
                "序号": i,
                "样本ID": r.get("sample_id", ""),
                "数据标签": label_cn(r.get("label")),
                "当前文本": r.get("current_caption", ""),
                "图像真实上下文": r.get("true_image_context", ""),
                "系统建议错配类型": cn_type(r.get("weak_mismatch_type", "")),
                "系统建议冲突字段": cn_fields(r.get("weak_conflict_fields", [])),
                "人工最终错配类型": cn_type(r.get("gold_mismatch_type", r.get("weak_mismatch_type", ""))),
                "人工最终冲突字段": cn_fields(r.get("gold_conflict_fields", r.get("weak_conflict_fields", []))),
                "人工理由": "请人工复核系统建议是否正确",
                "标注人": r.get("annotator", ""),
                "标注状态": "待复核",
                "split": r.get("split", ""),
                "domain": r.get("domain", ""),
                "image_id": r.get("image_id", ""),
                "text_id": r.get("text_id", ""),
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Chinese CSV files for manual attribution annotation.")
    ap.add_argument("--input", default="examples/attribution_eval_candidates.jsonl")
    ap.add_argument("--out-dir", default="examples")
    ap.add_argument("--split", type=int, default=2, help="Number of files to split into. Default: 2")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    out_dir = Path(args.out_dir)
    if args.split <= 1:
        out = out_dir / "annotation_中文.csv"
        write_csv(out, rows)
        print(out)
        return

    chunk = (len(rows) + args.split - 1) // args.split
    names = [chr(ord("A") + i) for i in range(args.split)]
    for i, name in enumerate(names):
        part = rows[i * chunk:(i + 1) * chunk]
        if not part:
            continue
        out = out_dir / f"annotation_{name}_中文.csv"
        write_csv(out, part)
        print(out)


if __name__ == "__main__":
    main()
