from __future__ import annotations

import argparse
import json
from pathlib import Path


def file_examples(root: Path, patterns, limit=10):
    out = []
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            out.append({"path": str(p), "size_bytes": p.stat().st_size})
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose local NewsCLIPpings and VisualNews metadata inputs.")
    ap.add_argument("--newsclippings-data-dir", required=True)
    ap.add_argument("--visualnews-metadata-dir", required=True)
    ap.add_argument("--output", default="outputs/input_check.json")
    args = ap.parse_args()

    nc = Path(args.newsclippings_data_dir)
    vn = Path(args.visualnews_metadata_dir)
    report = {
        "newsclippings_data_dir": str(nc),
        "newsclippings_exists": nc.exists(),
        "newsclippings_json_count": len(list(nc.rglob("*.json"))) if nc.exists() else 0,
        "newsclippings_json_examples": file_examples(nc, ["*.json"], limit=12) if nc.exists() else [],
        "visualnews_metadata_dir": str(vn),
        "visualnews_exists": vn.exists(),
        "visualnews_pickle_count": len(list(vn.rglob("*.p"))) + len(list(vn.rglob("*.pkl"))) + len(list(vn.rglob("*.pickle"))) if vn.exists() else 0,
        "visualnews_pickle_examples": file_examples(vn, ["*.p", "*.pkl", "*.pickle"], limit=12) if vn.exists() else [],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
