from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

MISMATCH_TYPES = [
    "entity mismatch", "location mismatch", "temporal mismatch",
    "event-type mismatch", "relation mismatch", "context omission",
    "benign illustrative image", "uncertain / evidence insufficient",
]

@dataclass
class EventTuple:
    source: str
    raw_text: str = ""
    entities: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PredictionResult:
    label: str
    confidence: float
    mismatch_type: str
    conflict_fields: List[str]
    event_scores: Dict[str, float]
    text_event: EventTuple
    image_event: EventTuple
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    model_version: str = "e3-vdt-demo-heuristic-v0.1"
    classification_policy: str = "event_sidecar_demo"
    decision_source: str = "event_heuristic_demo"
    baseline_label: Optional[str] = None
    baseline_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["confidence"] = round(float(self.confidence), 4)
        if self.baseline_score is not None:
            obj["baseline_score"] = round(float(self.baseline_score), 4)
        obj["event_scores"] = {k: round(float(v), 4) for k, v in self.event_scores.items()}
        return obj
