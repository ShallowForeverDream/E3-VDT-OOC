from __future__ import annotations
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple
from e3vdt.schemas import EventTuple

def _tokens(items: Iterable[str]) -> List[str]:
    return [str(x).strip().lower() for x in items if str(x).strip()]
def _best_string_sim(a: str, b: str) -> float:
    a=a.lower().strip(); b=b.lower().strip()
    if not a or not b: return 0.0
    if a == b or a in b or b in a: return 1.0
    return SequenceMatcher(None, a, b).ratio()
def list_similarity(left: List[str], right: List[str]) -> float:
    left=_tokens(left); right=_tokens(right)
    if not left and not right: return 0.5
    if not left or not right: return 0.25
    best=[]
    for x in left: best.append(max(_best_string_sim(x,y) for y in right))
    for y in right: best.append(max(_best_string_sim(y,x) for x in left))
    return max(0.0, min(1.0, sum(best)/len(best)))
def compute_event_scores(text_event: EventTuple, image_event: EventTuple) -> Dict[str,float]:
    return {
        "entity": list_similarity(text_event.entities, image_event.entities),
        "location": list_similarity(text_event.locations, image_event.locations),
        "time": list_similarity(text_event.times, image_event.times),
        "event_type": list_similarity(text_event.event_types, image_event.event_types),
        "relation": list_similarity(text_event.relations, image_event.relations),
    }
def summarize_scores(scores: Dict[str,float]) -> Tuple[float,List[str]]:
    values=list(scores.values()); overall=sum(values)/len(values) if values else 0.5
    conflicts=[k for k,v in scores.items() if v < 0.40]
    return overall, conflicts
