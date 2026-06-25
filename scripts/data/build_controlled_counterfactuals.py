from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
TITLE_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b")

LOCATION_LABELS = {"GPE", "LOC", "FAC"}
ENTITY_LABELS = {"PERSON", "ORG", "NORP"}


@dataclass(frozen=True)
class Span:
    field: str
    text: str
    start: int
    end: int
    subtype: str = ""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def is_non_ooc(row: Dict[str, Any]) -> bool:
    label = row.get("label")
    if label == 0:
        return True
    if isinstance(label, bool):
        return not label
    if isinstance(label, str):
        return label.strip().lower() in {"0", "false", "non-ooc", "non_ooc", "real", "match", "matched"}
    return False


def load_spacy(model: str):
    if spacy is None:
        return None
    try:
        return spacy.load(model)
    except Exception:
        return None


def add_span(out: List[Span], seen: set, field: str, text: str, start: int, end: int, subtype: str = "") -> None:
    text = clean_space(text)
    if not text or end <= start:
        return
    key = (field, start, end, text.lower())
    if key in seen:
        return
    # Avoid one-character and mostly punctuation spans.
    if len(re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", text)) < 2:
        return
    seen.add(key)
    out.append(Span(field=field, text=text, start=start, end=end, subtype=subtype))


def extract_spans(text: str, nlp=None) -> List[Span]:
    text = str(text or "")
    out: List[Span] = []
    seen = set()

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in LOCATION_LABELS:
                add_span(out, seen, "location", ent.text, ent.start_char, ent.end_char, ent.label_)
            elif ent.label_ in ENTITY_LABELS:
                add_span(out, seen, "entity", ent.text, ent.start_char, ent.end_char, ent.label_)
            elif ent.label_ in {"DATE", "TIME"} and YEAR_RE.search(ent.text):
                # First version keeps years only to avoid noisy date rewrites.
                for m in YEAR_RE.finditer(ent.text):
                    add_span(out, seen, "time", m.group(0), ent.start_char + m.start(), ent.start_char + m.end(), "YEAR")

    for m in YEAR_RE.finditer(text):
        add_span(out, seen, "time", m.group(0), m.start(), m.end(), "YEAR")

    # Conservative fallback entity spans.  This supplements machines without
    # spaCy and also recovers common headline-style names.
    if not any(s.field == "entity" for s in out):
        for m in TITLE_PHRASE_RE.finditer(text):
            phrase = m.group(0)
            if phrase.lower() in {"the", "a", "an"}:
                continue
            add_span(out, seen, "entity", phrase, m.start(), m.end(), "TITLE_PHRASE")

    return sorted(out, key=lambda s: (s.start, s.end))


def replacement_pool(non_ooc_rows: List[Dict[str, Any]], nlp=None) -> Dict[str, Dict[str, List[str]]]:
    pools: Dict[str, Dict[str, List[str]]] = {
        "location": defaultdict(list),
        "time": defaultdict(list),
        "entity": defaultdict(list),
    }
    seen: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in non_ooc_rows:
        text = row.get("current_caption") or row.get("true_image_context") or ""
        for sp in extract_spans(text, nlp=nlp):
            val = clean_space(sp.text)
            key = val.lower()
            if not key:
                continue
            subtype = sp.subtype or "ANY"
            if key not in seen[(sp.field, subtype)]:
                pools[sp.field][subtype].append(val)
                seen[(sp.field, subtype)].add(key)
            if key not in seen[(sp.field, "ANY")]:
                pools[sp.field]["ANY"].append(val)
                seen[(sp.field, "ANY")].add(key)
    return {f: {k: v for k, v in d.items() if v} for f, d in pools.items()}


def choose_replacement(pool: Dict[str, List[str]], span: Span, rng: random.Random) -> Optional[str]:
    candidates = list(pool.get(span.subtype, [])) or list(pool.get("ANY", []))
    candidates = [x for x in candidates if x.strip().lower() != span.text.strip().lower()]
    if not candidates:
        return None
    if span.field == "time":
        # Prefer a close but different year to preserve grammaticality.
        try:
            y = int(span.text)
            nearby = [str(x) for x in [y - 5, y - 3, y - 1, y + 1, y + 3, y + 5] if 1900 <= x <= 2035 and str(x) != span.text]
            if nearby:
                return rng.choice(nearby)
        except Exception:
            pass
    return rng.choice(candidates)


def replace_span(text: str, span: Span, repl: str) -> str:
    return text[: span.start] + repl + text[span.end :]


def validation_for(original: str, edited: str, span: Span, repl: str) -> Dict[str, Any]:
    return {
        "target_field_changed": span.text.strip().lower() != repl.strip().lower() and original != edited,
        "replacement_type_valid": True,
        "single_span_edited": True,
        "original_value_absent_at_span": edited[span.start : span.start + len(repl)] == repl,
        "original_text": span.text,
        "replacement_text": repl,
        "span_start": span.start,
        "span_end": span.end,
        "span_subtype": span.subtype,
    }


def make_positive(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    out = dict(row)
    base_id = str(row.get("sample_id") or row.get("id") or idx)
    out.update({
        "sample_id": f"{base_id}__cf_none",
        "source_sample_id": base_id,
        "label": 0,
        "gold_mismatch_type": "benign illustrative image",
        "gold_conflict_fields": [],
        "edit_type": "none",
        "edited_field": "none",
        "counterfactual_source": "controlled_counterfactual_from_non_ooc",
        "validation": {"positive_from_non_ooc": True},
    })
    return out


def make_counterfactual(row: Dict[str, Any], span: Span, replacement: str, idx: int) -> Dict[str, Any]:
    original = str(row.get("current_caption") or "")
    edited = replace_span(original, span, replacement)
    type_map = {
        "location": ("location mismatch", "location_swap"),
        "time": ("temporal mismatch", "time_swap"),
        "entity": ("entity mismatch", "entity_swap"),
    }
    mismatch_type, edit_type = type_map[span.field]
    base_id = str(row.get("sample_id") or row.get("id") or idx)
    out = dict(row)
    out.update({
        "sample_id": f"{base_id}__cf_{span.field}_{idx}",
        "source_sample_id": base_id,
        "original_caption": original,
        "current_caption": edited,
        "label": 1,
        "gold_mismatch_type": mismatch_type,
        "gold_conflict_fields": [span.field],
        "edit_type": edit_type,
        "edited_field": span.field,
        "edited_span_text": span.text,
        "replacement_text": replacement,
        "counterfactual_source": "controlled_counterfactual_from_non_ooc",
        "validation": validation_for(original, edited, span, replacement),
    })
    return out


def split_by_type(rows: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("edit_type", "unknown"))].append(r)
    out = {"train": [], "val": [], "test": []}
    for _, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train : n_train + n_val])
        out["test"].extend(group[n_train + n_val :])
    for rows2 in out.values():
        rng.shuffle(rows2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build controlled counterfactual attribution data from Non-OOC context pairs.")
    ap.add_argument("--context-pairs", default="outputs/cove_lite_context_pairs.jsonl")
    ap.add_argument("--output-dir", default="outputs/counterfactual")
    ap.add_argument("--max-per-type", type=int, default=200, help="Max rows per type: none/location/time/entity.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.context_pairs))
    non_ooc = [r for r in rows if is_non_ooc(r)]
    rng.shuffle(non_ooc)
    nlp = load_spacy(args.spacy_model)
    pools = replacement_pool(non_ooc, nlp=nlp)

    all_out: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input": args.context_pairs,
        "total_input": len(rows),
        "non_ooc_input": len(non_ooc),
        "max_per_type": args.max_per_type,
        "spacy_model_loaded": nlp is not None,
        "pool_sizes": {field: {k: len(v) for k, v in pool.items()} for field, pool in pools.items()},
        "attempted": Counter(),
        "kept": Counter(),
        "skip_reasons": Counter(),
    }

    # Positive rows.
    for i, row in enumerate(non_ooc[: args.max_per_type]):
        all_out.append(make_positive(row, i))
        stats["kept"]["none"] += 1

    targets = ["location", "time", "entity"]
    target_counts = Counter()
    for i, row in enumerate(non_ooc):
        text = str(row.get("current_caption") or "")
        spans = extract_spans(text, nlp=nlp)
        for target in targets:
            if target_counts[target] >= args.max_per_type:
                continue
            target_spans = [s for s in spans if s.field == target]
            if not target_spans:
                stats["skip_reasons"][f"{target}:no_span"] += 1
                continue
            span = rng.choice(target_spans)
            repl = choose_replacement(pools.get(target, {}), span, rng)
            stats["attempted"][f"{target}_swap"] += 1
            if not repl:
                stats["skip_reasons"][f"{target}:no_replacement"] += 1
                continue
            cf = make_counterfactual(row, span, repl, i)
            if not cf["validation"]["target_field_changed"]:
                stats["skip_reasons"][f"{target}:unchanged"] += 1
                continue
            all_out.append(cf)
            stats["kept"][f"{target}_swap"] += 1
            target_counts[target] += 1
        if all(target_counts[t] >= args.max_per_type for t in targets):
            break

    splits = split_by_type(all_out, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "controlled_counterfactual_all.jsonl", all_out)
    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"controlled_counterfactual_{split}.jsonl", split_rows)

    # Convert Counters for JSON.
    stats["attempted"] = dict(stats["attempted"])
    stats["kept"] = dict(stats["kept"])
    stats["skip_reasons"] = dict(stats["skip_reasons"])
    stats["split_counts"] = {k: len(v) for k, v in splits.items()}
    stats["type_counts"] = dict(Counter(r.get("edit_type") for r in all_out))
    stats["keep_rate"] = {
        k: stats["kept"].get(k, 0) / max(1, stats["attempted"].get(k, stats["kept"].get(k, 0)))
        for k in ["location_swap", "time_swap", "entity_swap"]
    }
    (out_dir / "counterfactual_generation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "records": len(all_out), "split_counts": stats["split_counts"], "type_counts": stats["type_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
