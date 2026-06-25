from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _s(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _best(x: str, ys: List[str]) -> float:
    return max([_s(x, y) for y in ys] or [0.0])


def _triples(event: Dict[str, Any]) -> List[Dict[str, str]]:
    triples = event.get("relations_structured") or []
    out = []
    for t in triples:
        if isinstance(t, dict):
            out.append({"subject": str(t.get("subject", "")), "predicate": str(t.get("predicate", "")), "object": str(t.get("object", ""))})
    return out


def align_event_graphs(current_event: Dict[str, Any], true_event: Dict[str, Any], conflict_threshold: float = 0.35) -> Dict[str, Any]:
    """Lightweight graph alignment inspired by evidence-graph methods.

    It does not train a GNN. It aligns structured triples and event fields to
    produce explainable graph conflicts that can be compared to field-wise NLI.
    """
    cur_triples = _triples(current_event)
    true_triples = _triples(true_event)
    aligned_edges: List[Dict[str, Any]] = []
    for ct in cur_triples:
        best = None
        best_score = -1.0
        for tt in true_triples:
            score = (_s(ct["subject"], tt["subject"]) + _s(ct["predicate"], tt["predicate"]) + _s(ct["object"], tt["object"])) / 3.0
            if score > best_score:
                best = tt
                best_score = score
        if best is not None:
            aligned_edges.append({"current": ct, "true": best, "score": round(best_score, 6)})
    graph_conflicts: List[str] = []
    # If subjects/predicates align but objects diverge, relation/object grounding is suspect.
    for e in aligned_edges:
        c = e["current"]
        t = e["true"]
        subj = _s(c["subject"], t["subject"])
        pred = _s(c["predicate"], t["predicate"])
        obj = _s(c["object"], t["object"])
        if subj >= 0.65 and pred >= 0.45 and obj < conflict_threshold:
            graph_conflicts.append("relation")
    score = sum(e["score"] for e in aligned_edges) / len(aligned_edges) if aligned_edges else 0.0
    return {
        "graph_alignment_score": round(score, 6),
        "graph_conflicts": sorted(set(graph_conflicts)),
        "aligned_edges": aligned_edges,
        "num_current_edges": len(cur_triples),
        "num_true_edges": len(true_triples),
    }
