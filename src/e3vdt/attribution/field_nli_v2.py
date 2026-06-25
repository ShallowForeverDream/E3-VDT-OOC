from __future__ import annotations

from typing import Any, Dict, List, Optional

FIELD_TO_TYPE = {
    "entity": "entity mismatch",
    "location": "location mismatch",
    "time": "temporal mismatch",
    "event_type": "event-type mismatch",
    "relation": "relation mismatch",
}


def _join(xs: List[str]) -> str:
    return ", ".join([str(x) for x in xs if str(x).strip()])


def build_field_hypothesis(field: str, values: List[str]) -> Optional[str]:
    vals = _join(values)
    if not vals:
        return None
    if field == "entity":
        return f"The image event involves {vals}."
    if field == "location":
        return f"The image event took place in {vals}."
    if field == "time":
        return f"The image event happened at {vals}."
    if field == "event_type":
        return f"The image event is about {vals}."
    if field == "relation":
        return f"The image event includes the relation or action: {vals}."
    return f"The image event includes {vals}."


class NLIPredictor:
    """Lazy HuggingFace NLI wrapper with a deterministic lexical fallback."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1, use_transformers: bool = True) -> None:
        self.model_name = model_name
        self.device = device
        self.use_transformers = use_transformers
        self._pipe = None
        self._load_error: Optional[str] = None

    def _ensure_pipe(self):
        if not self.use_transformers:
            return None
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline("text-classification", model=self.model_name, device=self.device, return_all_scores=True)
        except Exception as e:  # pragma: no cover
            self._load_error = repr(e)
            self._pipe = None
        return self._pipe

    @staticmethod
    def _fallback(premise: str, hypothesis: str) -> Dict[str, float]:
        p = set(premise.lower().replace(".", " ").replace(",", " ").split())
        h = set(hypothesis.lower().replace(".", " ").replace(",", " ").split())
        if not h:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        overlap = len(p & h) / max(1, len(h))
        entail = max(0.0, min(0.85, overlap))
        contradiction = 0.15 if overlap < 0.15 else 0.05
        neutral = max(0.0, 1.0 - entail - contradiction)
        return {"entailment": entail, "neutral": neutral, "contradiction": contradiction}

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        pipe = self._ensure_pipe()
        if pipe is None:
            out = self._fallback(premise, hypothesis)
            out["_backend"] = "lexical_fallback"
            if self._load_error:
                out["_load_error"] = self._load_error
            return out
        try:
            res = pipe({"text": premise, "text_pair": hypothesis})
            if res and isinstance(res[0], list):
                res = res[0]
            scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
            for item in res:
                lab = str(item.get("label", "")).lower()
                val = float(item.get("score", 0.0))
                if "entail" in lab:
                    scores["entailment"] = val
                elif "contrad" in lab:
                    scores["contradiction"] = val
                elif "neutral" in lab:
                    scores["neutral"] = val
            scores["_backend"] = self.model_name
            return scores
        except Exception as e:  # pragma: no cover
            out = self._fallback(premise, hypothesis)
            out["_backend"] = "lexical_fallback_after_error"
            out["_error"] = repr(e)
            return out


def run_field_nli(
    current_event: Dict[str, Any],
    true_event: Dict[str, Any],
    true_context: str,
    nli: NLIPredictor,
    contradiction_threshold: float = 0.60,
    entailment_threshold: float = 0.60,
) -> Dict[str, Any]:
    fields = {
        "entity": current_event.get("entities", []),
        "location": current_event.get("locations", []),
        "time": current_event.get("times", []),
        "event_type": current_event.get("event_types", []),
        "relation": current_event.get("relations", []),
    }
    field_nli: Dict[str, Any] = {}
    conflict_fields: List[str] = []
    for field, vals in fields.items():
        hyp = build_field_hypothesis(field, vals)
        if not hyp:
            field_nli[field] = {"label": "missing", "scores": {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}, "hypothesis": None}
            continue
        scores = nli.score(true_context or "", hyp)
        c = float(scores.get("contradiction", 0.0))
        e = float(scores.get("entailment", 0.0))
        if c >= contradiction_threshold:
            label = "contradiction"
            conflict_fields.append(field)
        elif e >= entailment_threshold:
            label = "entailment"
        else:
            label = "neutral"
        field_nli[field] = {"label": label, "scores": scores, "hypothesis": hyp}
    if conflict_fields:
        primary = max(conflict_fields, key=lambda f: float(field_nli[f]["scores"].get("contradiction", 0.0)))
        mismatch_type = FIELD_TO_TYPE.get(primary, "context omission")
    else:
        mismatch_type = "context omission" if any(field_nli[f]["label"] == "neutral" for f in field_nli) else "benign illustrative image"
    return {"field_nli": field_nli, "conflict_fields": conflict_fields, "mismatch_type": mismatch_type}
