from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from e3vdt.event.extractor import extract_event_tuple
from e3vdt.event.normalize import normalize_event_dict, clean_text

try:
    import dateparser  # type: ignore
except Exception:  # pragma: no cover
    dateparser = None

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None

_NLP_CACHE: Dict[str, Any] = {}


def _load_spacy(model: str):
    if spacy is None:
        return None
    if model not in _NLP_CACHE:
        try:
            _NLP_CACHE[model] = spacy.load(model)
        except Exception:
            _NLP_CACHE[model] = None
    return _NLP_CACHE[model]


def _merge_unique(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in list(a or []) + list(b or []):
        s = clean_text(str(x))
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def extract_with_spacy(text: str, model: str = "en_core_web_sm") -> Dict[str, List[str]]:
    """Optional pretrained IE layer.

    If spaCy/model is unavailable, this function returns empty fields instead of
    failing. This keeps the pipeline reproducible on machines that have not yet
    installed optional IE dependencies.
    """
    nlp = _load_spacy(model)
    if nlp is None:
        return {"entities": [], "locations": [], "times": []}
    doc = nlp(text or "")
    entities: List[str] = []
    locations: List[str] = []
    times: List[str] = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "NORP"}:
            entities.append(ent.text)
        elif ent.label_ in {"GPE", "LOC", "FAC"}:
            locations.append(ent.text)
        elif ent.label_ in {"DATE", "TIME"}:
            times.append(ent.text)
    return {"entities": entities, "locations": locations, "times": times}


def extract_openie_like_triples(text: str) -> List[Dict[str, str]]:
    """A deterministic fallback relation extractor.

    This is not full OpenIE. It is a local, auditable fallback that turns common
    news-style clauses into subject-predicate-object triples. It remains a
    baseline until a full OpenIE/SRL system is installed.
    """
    text = clean_text(text)
    triples: List[Dict[str, str]] = []
    patterns = [
        r"(?P<subj>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}|[Pp]olice|[Pp]rotesters|[Oo]fficials|[Ff]ans|[Pp]eople)\s+(?P<pred>attacked|met|visited|arrested|protested|gathered|marched|won|beat|rescued|evacuated|said|announced)\s+(?P<obj>[^.;,]{2,80})",
        r"(?P<subj>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}|[Pp]rotesters|[Pp]olice)\s+(?P<pred>in|at|near)\s+(?P<obj>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            triples.append({
                "subject": clean_text(m.group("subj")),
                "predicate": clean_text(m.group("pred")),
                "object": clean_text(m.group("obj")),
            })
    return triples


def event_from_text(text: str, extractor: str = "enhanced", spacy_model: str = "en_core_web_sm") -> Dict[str, Any]:
    """Return an event tuple dict.

    extractor values:
    - rule: current repository rule baseline only.
    - enhanced: rule baseline + optional spaCy NER + fallback triples.
    - spacy: optional spaCy NER + rule event type/relation keywords.
    """
    base_obj = extract_event_tuple(text, source="caption_or_context")
    base = normalize_event_dict(getattr(base_obj, "__dict__", {}))
    if extractor == "rule":
        base["extractor"] = "rule"
        base["relations_structured"] = []
        return base

    sp = extract_with_spacy(text, model=spacy_model)
    triples = extract_openie_like_triples(text)
    merged = dict(base)
    merged["entities"] = _merge_unique(base.get("entities", []), sp.get("entities", []))
    merged["locations"] = _merge_unique(base.get("locations", []), sp.get("locations", []))
    merged["times"] = _merge_unique(base.get("times", []), sp.get("times", []))
    # Keep existing keyword relations and add predicate labels from triples.
    triple_preds = [t.get("predicate", "") for t in triples]
    merged["relations"] = _merge_unique(base.get("relations", []), triple_preds)
    merged["relations_structured"] = triples
    merged["extractor"] = extractor
    return normalize_event_dict(merged) | {"relations_structured": triples, "extractor": extractor}


def dump_event_json(text: str, extractor: str = "enhanced") -> str:
    return json.dumps(event_from_text(text, extractor=extractor), ensure_ascii=False)
