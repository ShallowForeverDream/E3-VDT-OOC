from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, Iterable

from e3vdt.event.normalize import normalize_event_dict


def _sim(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    l = {str(x).strip().lower() for x in left if str(x).strip()}
    r = {str(x).strip().lower() for x in right if str(x).strip()}
    if not l and not r:
        return 0.5
    if not l or not r:
        return 0.0
    return len(l & r) / max(1, len(l | r))


def count_filled_fields(event: Dict[str, Any]) -> int:
    return sum(1 for k in ["entities", "locations", "times", "event_types", "relations"] if event.get(k))


def score_evidence_relevance(
    current_caption: str,
    true_context: str,
    current_event: Dict[str, Any],
    true_event: Dict[str, Any],
    min_context_chars: int = 20,
    sufficient_threshold: float = 0.25,
) -> Dict[str, Any]:
    current_event = normalize_event_dict(current_event)
    true_event = normalize_event_dict(true_event)
    text_similarity = _sim(current_caption, true_context)
    field_overlaps = {
        "entity": _overlap(current_event.get("entities", []), true_event.get("entities", [])),
        "location": _overlap(current_event.get("locations", []), true_event.get("locations", [])),
        "time": _overlap(current_event.get("times", []), true_event.get("times", [])),
        "event_type": _overlap(current_event.get("event_types", []), true_event.get("event_types", [])),
        "relation": _overlap(current_event.get("relations", []), true_event.get("relations", [])),
    }
    filled_true = count_filled_fields(true_event)
    filled_current = count_filled_fields(current_event)
    field_overlap_mean = sum(field_overlaps.values()) / len(field_overlaps)
    context_len_score = min(1.0, len(true_context or "") / max(1, min_context_chars * 3))
    information_score = min(1.0, filled_true / 3.0)
    relevance = 0.35 * text_similarity + 0.35 * field_overlap_mean + 0.15 * context_len_score + 0.15 * information_score
    sufficient = bool(true_context and len(true_context) >= min_context_chars and filled_true >= 1 and relevance >= sufficient_threshold)
    reason = "sufficient"
    if not true_context or len(true_context) < min_context_chars:
        reason = "true_context_too_short"
    elif filled_true < 1:
        reason = "true_context_has_no_extracted_fields"
    elif relevance < sufficient_threshold:
        reason = "low_relevance"
    return {
        "evidence_relevance": round(float(relevance), 6),
        "evidence_sufficiency": "sufficient" if sufficient else "insufficient",
        "evidence_reason": reason,
        "text_similarity": round(float(text_similarity), 6),
        "field_overlaps": field_overlaps,
        "filled_true_fields": filled_true,
        "filled_current_fields": filled_current,
        "context_length": len(true_context or ""),
    }
