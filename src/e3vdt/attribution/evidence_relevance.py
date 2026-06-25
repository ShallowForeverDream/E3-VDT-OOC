from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Set

from e3vdt.schemas import EventTuple
from e3vdt.event.consistency import compute_event_scores


@dataclass
class EvidenceRelevanceResult:
    score: float
    level: str
    sufficient: bool
    token_overlap: float
    event_overlap: float
    context_length: int
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        obj = asdict(self)
        obj["score"] = round(float(self.score), 4)
        obj["token_overlap"] = round(float(self.token_overlap), 4)
        obj["event_overlap"] = round(float(self.event_overlap), 4)
        return obj


def _tokens(text: str) -> Set[str]:
    words = re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fa5]{2,}", text or "")
    stop = {"the", "and", "with", "for", "from", "that", "this", "during", "after", "before", "will", "were", "was", "are"}
    return {w.lower() for w in words if w.lower() not in stop}


def _non_empty_fields(event: EventTuple) -> int:
    return sum(bool(x) for x in [event.entities, event.locations, event.times, event.event_types, event.relations])


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 0.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def score_evidence_relevance(current_caption: str, true_context: str, current_event: EventTuple, true_event: EventTuple) -> EvidenceRelevanceResult:
    """RED-DOT/CMIE-inspired lightweight evidence sufficiency scorer.

    This is not a learned evidence selector. It is a deterministic system-demo
    implementation that exposes the signals we will later replace/validate with
    a trained or NLI-based relevance model.
    """
    context_len = len((true_context or "").split())
    token_overlap = _jaccard(_tokens(current_caption), _tokens(true_context))
    scores = compute_event_scores(current_event, true_event)
    # Ignore 0.5 unknown/unknown scores when estimating evidence overlap.
    informative = [v for v in scores.values() if abs(v - 0.5) > 1e-6]
    event_overlap = sum(informative) / len(informative) if informative else 0.0
    field_count = _non_empty_fields(true_event)

    length_score = min(1.0, context_len / 10.0)
    field_score = min(1.0, field_count / 2.0)
    score = 0.35 * token_overlap + 0.35 * event_overlap + 0.20 * field_score + 0.10 * length_score

    sufficient = bool(true_context.strip()) and (field_count >= 1 or context_len >= 6)
    if not sufficient:
        level = "insufficient"
        rationale = "true image context 为空或缺少可抽取事件字段，系统不强行归因。"
    elif score >= 0.55:
        level = "high"
        rationale = "true context 与 current caption 有足够语义/事件字段关联，可用于字段归因。"
    elif score >= 0.30:
        level = "medium"
        rationale = "true context 提供部分相关证据，归因结果需要人工复核。"
    else:
        level = "low"
        rationale = "true context 与 current caption 关联较弱，可能是跨主题错配或证据不足。"

    return EvidenceRelevanceResult(
        score=max(0.0, min(1.0, score)),
        level=level,
        sufficient=sufficient,
        token_overlap=token_overlap,
        event_overlap=event_overlap,
        context_length=context_len,
        rationale=rationale,
    )
