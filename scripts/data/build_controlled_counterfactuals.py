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
DEFAULT_YEAR_POOL = [str(y) for y in range(2016, 2027)]
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "this", "that",
    "these", "those", "it", "its", "into", "after", "before", "during", "over", "under",
    "across", "about", "more", "than", "not", "no", "but", "his", "her", "their", "our",
    "your", "who", "which", "will", "would", "can", "could", "should", "may", "might",
}

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


def load_exclusion_sets(paths: Iterable[str]) -> Dict[str, set]:
    """Load sample/text/image ids that must not be reused for original-OOC training.

    The manual real-OOC gold set is for evaluation, not training.  When we add
    original OOC rows as `different-event mismatch`, any overlap with manual
    samples must be filtered out to avoid evaluation leakage.
    """
    out = {"sample_id": set(), "text_id": set(), "image_id": set(), "source_sample_id": set()}
    for raw in paths:
        if not str(raw or "").strip():
            continue
        path = Path(raw)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in list(out):
                val = str(row.get(key) or "").strip()
                if val:
                    out[key].add(val)
            sid = str(row.get("sample_id") or "").strip()
            if sid:
                out["source_sample_id"].add(sid)
    return out


def excluded_by_manual_gold(row: Dict[str, Any], exclude_sets: Dict[str, set]) -> bool:
    for key in ["sample_id", "text_id", "image_id", "source_sample_id"]:
        val = str(row.get(key) or "").strip()
        if val and val in exclude_sets.get(key, set()):
            return True
    return False


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text or "")))


def is_non_ooc(row: Dict[str, Any]) -> bool:
    label = row.get("label")
    if label == 0:
        return True
    if isinstance(label, bool):
        return not label
    if isinstance(label, str):
        return label.strip().lower() in {"0", "false", "non-ooc", "non_ooc", "real", "match", "matched"}
    return False


def is_ooc(row: Dict[str, Any]) -> bool:
    return not is_non_ooc(row)


def content_tokens(text: str) -> set:
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", str(text or ""))
        if len(w) > 1 and w.lower() not in STOPWORDS
    }


def token_jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


_ANNOTATION_CACHE: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}


def annotation_for_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Recover the original NewsCLIPpings annotation for similarity_score.

    cove_lite rows keep the source json path plus text/image ids, but not the
    annotation similarity.  Loading is cached per json file and only used for
    optional different-event filtering.
    """
    src = str(row.get("newsclippings_file") or "").strip()
    if not src:
        return None
    if src not in _ANNOTATION_CACHE:
        path = Path(src)
        if not path.exists():
            _ANNOTATION_CACHE[src] = {}
        else:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                _ANNOTATION_CACHE[src] = {
                    (str(a.get("id")), str(a.get("image_id"))): a
                    for a in obj.get("annotations", [])
                    if isinstance(a, dict)
                }
            except Exception:
                _ANNOTATION_CACHE[src] = {}
    text_id = str(row.get("text_id") or row.get("sample_id") or "").strip()
    image_id = str(row.get("image_id") or "").strip()
    return _ANNOTATION_CACHE[src].get((text_id, image_id))


def annotation_similarity(row: Dict[str, Any]) -> Optional[float]:
    ann = annotation_for_row(row)
    if not ann or ann.get("similarity_score") is None:
        return None
    try:
        return float(ann.get("similarity_score"))
    except Exception:
        return None


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
    # Time counterfactuals are intentionally constrained to YEAR -> YEAR.
    # Even when the sampled split has few distinct years, keep a stable
    # 2016-2026 pool so valid replacement does not become the bottleneck.
    for year in DEFAULT_YEAR_POOL:
        for key in ["YEAR", "ANY"]:
            if year.lower() not in seen[("time", key)]:
                pools["time"][key].append(year)
                seen[("time", key)].add(year.lower())
    return {f: {k: v for k, v in d.items() if v} for f, d in pools.items()}


def replacement_candidates(pool: Dict[str, List[str]], span: Span) -> List[str]:
    candidates = list(pool.get(span.subtype, [])) or list(pool.get("ANY", []))
    if span.field == "time":
        candidates = list(dict.fromkeys(candidates + list(pool.get("YEAR", [])) + DEFAULT_YEAR_POOL))
    candidates = [x for x in candidates if x.strip().lower() != span.text.strip().lower()]
    return list(dict.fromkeys(candidates))


def choose_replacements(pool: Dict[str, List[str]], span: Span, rng: random.Random, max_n: int = 1) -> List[str]:
    candidates = replacement_candidates(pool, span)
    if not candidates:
        return []
    if span.field == "time":
        # YEAR -> YEAR only.  Prefer a close but different year to preserve
        # grammaticality while staying inside the available/default pool.
        try:
            y = int(span.text)
            candidate_set = set(candidates)
            nearby = [
                str(x)
                for x in [y - 5, y - 3, y - 1, y + 1, y + 3, y + 5]
                if str(x) in candidate_set and str(x) != span.text
            ]
            if nearby:
                rng.shuffle(nearby)
                rest = [x for x in candidates if x not in set(nearby)]
                rng.shuffle(rest)
                return (nearby + rest)[:max_n]
        except Exception:
            pass
    rng.shuffle(candidates)
    return candidates[:max_n]


def choose_replacement(pool: Dict[str, List[str]], span: Span, rng: random.Random) -> Optional[str]:
    vals = choose_replacements(pool, span, rng, max_n=1)
    return vals[0] if vals else None


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


def make_counterfactual(row: Dict[str, Any], span: Span, replacement: str, idx: int, variant_idx: int = 0) -> Dict[str, Any]:
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
        "sample_id": f"{base_id}__cf_{span.field}_{idx}_{variant_idx}",
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


def make_original_ooc_different_event(row: Dict[str, Any], idx: int, sim: Optional[float], jac: float) -> Dict[str, Any]:
    base_id = str(row.get("sample_id") or row.get("id") or f"ooc_{idx}")
    out = dict(row)
    out.update({
        "sample_id": f"{base_id}__orig_ooc_different_event",
        "source_sample_id": f"orig_ooc:{base_id}",
        "label": 1,
        "gold_mismatch_type": "different-event mismatch",
        "gold_conflict_fields": ["entity", "location", "event_type"],
        "edit_type": "different_event_original_ooc",
        "edited_field": "different_event",
        "counterfactual_source": "original_ooc_filtered_low_similarity",
        "validation": {
            "original_ooc_falsified_pair": True,
            "newsclippings_similarity_score": sim,
            "caption_true_context_token_jaccard": jac,
            "selection_rule": "OOC and low annotation similarity / low token overlap",
        },
    })
    return out


def select_original_ooc_different_events(
    rows: List[Dict[str, Any]],
    nlp,
    rng: random.Random,
    max_n: int,
    max_similarity: float,
    max_token_jaccard: float,
    min_caption_tokens: int,
    exclude_sets: Optional[Dict[str, set]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "requested": max_n,
        "selected": 0,
        "max_similarity": max_similarity,
        "max_token_jaccard": max_token_jaccard,
        "min_caption_tokens": min_caption_tokens,
        "skip_reasons": Counter(),
        "similarity_bins": Counter(),
    }
    exclude_sets = exclude_sets or {}
    candidates = [r for r in rows if is_ooc(r)]
    rng.shuffle(candidates)
    for idx, row in enumerate(candidates):
        if len(selected) >= max_n:
            break
        if excluded_by_manual_gold(row, exclude_sets):
            stats["skip_reasons"]["manual_gold_overlap"] += 1
            continue
        cur = clean_space(row.get("current_caption") or "")
        true_ctx = clean_space(row.get("true_image_context") or "")
        if token_count(cur) < min_caption_tokens or token_count(true_ctx) < min_caption_tokens:
            stats["skip_reasons"]["too_short"] += 1
            continue
        jac = token_jaccard(cur, true_ctx)
        if jac > max_token_jaccard:
            stats["skip_reasons"]["token_overlap_too_high"] += 1
            continue
        sim = annotation_similarity(row)
        if sim is not None:
            bin_key = f"{int(sim * 10) / 10:.1f}-{int(sim * 10) / 10 + 0.1:.1f}"
            stats["similarity_bins"][bin_key] += 1
            if sim > max_similarity:
                stats["skip_reasons"]["similarity_too_high"] += 1
                continue
        selected.append(make_original_ooc_different_event(row, idx, sim, jac))
    stats["selected"] = len(selected)
    stats["skip_reasons"] = dict(stats["skip_reasons"])
    stats["similarity_bins"] = dict(stats["similarity_bins"])
    return selected, stats


def split_by_group(rows: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    """Group split that prevents source_sample_id/image_id/text_id leakage.

    We build connected components over all available identifiers.  If two rows
    share any identifier, they are assigned to the same split.
    """
    rng = random.Random(seed)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    row_nodes: List[List[str]] = []
    for idx, r in enumerate(rows):
        nodes = []
        for prefix, key in [("src", "source_sample_id"), ("img", "image_id"), ("txt", "text_id")]:
            val = str(r.get(key) or "").strip()
            if val:
                nodes.append(f"{prefix}:{val}")
        if not nodes:
            nodes = [f"row:{idx}"]
        for node in nodes[1:]:
            union(nodes[0], node)
        row_nodes.append(nodes)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, nodes in zip(rows, row_nodes):
        groups[find(nodes[0])].append(r)

    keys = list(groups)
    rng.shuffle(keys)
    n_groups = len(keys)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    split_keys = {
        "train": set(keys[:n_train]),
        "val": set(keys[n_train : n_train + n_val]),
        "test": set(keys[n_train + n_val :]),
    }
    out = {"train": [], "val": [], "test": []}
    for split, ks in split_keys.items():
        for k in ks:
            out[split].extend(groups[k])
        rng.shuffle(out[split])
    return out


def split_by_type(rows: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    """Legacy row-level stratified split kept for reproducibility comparisons.

    Do not use for final experiments because variants from one original sample
    can cross train/val/test.  The default path uses split_by_group().
    """
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
    ap.add_argument(
        "--max-time-variants-per-source",
        type=int,
        default=6,
        help="Allow multiple YEAR->YEAR variants from one source row for time_swap. Group split keeps variants together.",
    )
    ap.add_argument("--max-location-variants-per-source", type=int, default=2, help="Allow multiple location replacements from one source row when balancing large runs.")
    ap.add_argument("--max-entity-variants-per-source", type=int, default=1, help="Allow multiple entity replacements from one source row when balancing large runs.")
    ap.add_argument("--include-original-ooc-different-event", action="store_true", help="Add strictly filtered original OOC rows as different-event mismatch training examples.")
    ap.add_argument("--max-different-event", type=int, default=0, help="Max original OOC different-event rows; default 0 means max-per-type.")
    ap.add_argument("--different-event-max-similarity", type=float, default=0.65, help="Maximum NewsCLIPpings similarity_score for original OOC rows used as different-event.")
    ap.add_argument("--different-event-max-token-jaccard", type=float, default=0.08, help="Maximum token Jaccard between caption and true context for different-event rows.")
    ap.add_argument("--different-event-min-caption-tokens", type=int, default=4)
    ap.add_argument(
        "--exclude-different-event-gold",
        nargs="*",
        default=[],
        help="JSONL files whose sample_id/text_id/image_id must be excluded from original-OOC different-event training rows.",
    )
    ap.add_argument("--row-level-split", action="store_true", help="Legacy split; allows leakage and is only for ablation/debugging.")
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
        "max_time_variants_per_source": args.max_time_variants_per_source,
        "max_location_variants_per_source": args.max_location_variants_per_source,
        "max_entity_variants_per_source": args.max_entity_variants_per_source,
        "spacy_model_loaded": nlp is not None,
        "pool_sizes": {field: {k: len(v) for k, v in pool.items()} for field, pool in pools.items()},
        "default_year_pool": DEFAULT_YEAR_POOL,
        "attempted": Counter(),
        "kept": Counter(),
        "skip_reasons": Counter(),
        "span_rows": Counter(),
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
            if any(s.field == target for s in spans):
                stats["span_rows"][target] += 1
        for target in targets:
            if target_counts[target] >= args.max_per_type:
                continue
            target_spans = [s for s in spans if s.field == target]
            if not target_spans:
                stats["skip_reasons"][f"{target}:no_span"] += 1
                continue
            rng.shuffle(target_spans)
            row_variant_budget = {
                "time": args.max_time_variants_per_source,
                "location": args.max_location_variants_per_source,
                "entity": args.max_entity_variants_per_source,
            }.get(target, 1)
            row_variants_kept = 0
            row_any_replacement = False
            for span_idx, span in enumerate(target_spans):
                if target_counts[target] >= args.max_per_type or row_variants_kept >= row_variant_budget:
                    break
                needed = min(args.max_per_type - target_counts[target], row_variant_budget - row_variants_kept)
                repls = choose_replacements(pools.get(target, {}), span, rng, max_n=needed)
                if repls:
                    row_any_replacement = True
                for local_variant_idx, repl in enumerate(repls):
                    if target_counts[target] >= args.max_per_type or row_variants_kept >= row_variant_budget:
                        break
                    stats["attempted"][f"{target}_swap"] += 1
                    variant_id = row_variants_kept * 100 + span_idx * 10 + local_variant_idx
                    cf = make_counterfactual(row, span, repl, i, variant_idx=variant_id)
                    if not cf["validation"]["target_field_changed"]:
                        stats["skip_reasons"][f"{target}:unchanged"] += 1
                        continue
                    all_out.append(cf)
                    stats["kept"][f"{target}_swap"] += 1
                    target_counts[target] += 1
                    row_variants_kept += 1
            if not row_any_replacement:
                stats["skip_reasons"][f"{target}:no_replacement"] += 1
        if all(target_counts[t] >= args.max_per_type for t in targets):
            break

    if args.include_original_ooc_different_event:
        max_diff = args.max_different_event if args.max_different_event > 0 else args.max_per_type
        exclude_sets = load_exclusion_sets(args.exclude_different_event_gold)
        diff_rows, diff_stats = select_original_ooc_different_events(
            rows=rows,
            nlp=nlp,
            rng=rng,
            max_n=max_diff,
            max_similarity=args.different_event_max_similarity,
            max_token_jaccard=args.different_event_max_token_jaccard,
            min_caption_tokens=args.different_event_min_caption_tokens,
            exclude_sets=exclude_sets,
        )
        diff_stats["exclude_gold_files"] = list(args.exclude_different_event_gold)
        diff_stats["exclude_set_sizes"] = {k: len(v) for k, v in exclude_sets.items()}
        all_out.extend(diff_rows)
        stats["kept"]["different_event_original_ooc"] += len(diff_rows)
        stats["attempted"]["different_event_original_ooc"] += diff_stats.get("selected", 0) + sum(diff_stats.get("skip_reasons", {}).values())
        stats["different_event_original_ooc"] = diff_stats

    splits = split_by_type(all_out, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio) if args.row_level_split else split_by_group(all_out, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "controlled_counterfactual_all.jsonl", all_out)
    for split, split_rows in splits.items():
        write_jsonl(out_dir / f"controlled_counterfactual_{split}.jsonl", split_rows)

    # Convert Counters for JSON.
    stats["attempted"] = dict(stats["attempted"])
    stats["kept"] = dict(stats["kept"])
    stats["skip_reasons"] = dict(stats["skip_reasons"])
    stats["span_rows"] = dict(stats["span_rows"])
    stats["split_counts"] = {k: len(v) for k, v in splits.items()}
    stats["split_strategy"] = "row_level_by_type_legacy" if args.row_level_split else "group_by_source_sample_id_image_id"
    stats["type_counts"] = dict(Counter(r.get("edit_type") for r in all_out))
    stats["target_counts"] = dict(target_counts)
    stats["split_type_counts"] = {split: dict(Counter(r.get("edit_type") for r in split_rows)) for split, split_rows in splits.items()}
    stats["keep_rate"] = {
        k: stats["kept"].get(k, 0) / max(1, stats["attempted"].get(k, stats["kept"].get(k, 0)))
        for k in ["location_swap", "time_swap", "entity_swap"]
    }
    stats["time_swap_summary"] = {
        "generated": stats["attempted"].get("time_swap", 0),
        "kept": stats["kept"].get("time_swap", 0),
        "skipped_no_span": stats["skip_reasons"].get("time:no_span", 0),
        "skipped_no_replacement": stats["skip_reasons"].get("time:no_replacement", 0),
        "skipped_unchanged": stats["skip_reasons"].get("time:unchanged", 0),
        "note": "time_swap edits are YYYY -> YYYY; if kept is below max_per_type, the limiting factor is available current_caption rows with a YYYY span.",
    }
    (out_dir / "counterfactual_generation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "records": len(all_out), "split_counts": stats["split_counts"], "type_counts": stats["type_counts"], "time_swap_summary": stats["time_swap_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
