from __future__ import annotations

import re
from typing import Iterable, List, Dict, Any

LOCATION_ALIASES = {
    "u.s.": "united states",
    "us": "united states",
    "usa": "united states",
    "u.s.a.": "united states",
    "america": "united states",
    "uk": "united kingdom",
    "britain": "united kingdom",
    "england": "united kingdom",
    "washington dc": "washington",
    "washington d.c.": "washington",
    "new york city": "new york",
}

EVENT_ALIASES = {
    "war": "war/conflict",
    "conflict": "war/conflict",
    "attack": "war/conflict",
    "strike": "war/conflict",
    "demonstration": "protest",
    "rally": "protest",
    "march": "protest",
    "vote": "politics",
    "election": "politics",
    "trial": "court/crime",
    "arrest": "court/crime",
    "match": "sports",
    "game": "sports",
}


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def norm_token(x: str) -> str:
    x = clean_text(str(x)).strip(".,;:!?，。；：！？'\"()[]{}").lower()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_location(x: str) -> str:
    key = norm_token(x)
    return LOCATION_ALIASES.get(key, key)


def normalize_event_type(x: str) -> str:
    key = norm_token(x)
    return EVENT_ALIASES.get(key, key)


def normalize_list(items: Iterable[str], kind: str = "generic") -> List[str]:
    out = []
    seen = set()
    for item in items or []:
        if kind == "location":
            val = normalize_location(str(item))
        elif kind == "event_type":
            val = normalize_event_type(str(item))
        else:
            val = norm_token(str(item))
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def event_tuple_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"entities": [], "locations": [], "times": [], "event_types": [], "relations": []}
    if isinstance(obj, dict):
        data = dict(obj)
    elif hasattr(obj, "to_dict"):
        data = obj.to_dict()
    else:
        data = getattr(obj, "__dict__", {})
    return {
        "entities": list(data.get("entities", []) or []),
        "locations": list(data.get("locations", []) or []),
        "times": list(data.get("times", []) or []),
        "event_types": list(data.get("event_types", []) or []),
        "relations": list(data.get("relations", []) or []),
        "raw_text": data.get("raw_text", ""),
        "source": data.get("source", ""),
    }


def normalize_event_dict(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entities": normalize_list(event.get("entities", []), "entity"),
        "locations": normalize_list(event.get("locations", []), "location"),
        "times": normalize_list(event.get("times", []), "time"),
        "event_types": normalize_list(event.get("event_types", []), "event_type"),
        "relations": normalize_list(event.get("relations", []), "relation"),
        "raw_text": event.get("raw_text", ""),
        "source": event.get("source", ""),
    }
