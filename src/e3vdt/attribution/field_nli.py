from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from e3vdt.schemas import EventTuple
from e3vdt.event.consistency import list_similarity

FIELD_MAP = {
    "entity": ("entities", "主体/实体"),
    "location": ("locations", "地点"),
    "time": ("times", "时间"),
    "event_type": ("event_types", "事件类型"),
    "relation": ("relations", "动作/关系"),
}


@dataclass
class FieldNLIResult:
    field: str
    label: str
    confidence: float
    current_values: List[str]
    true_values: List[str]
    score: float
    hypothesis: str
    evidence: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        obj = asdict(self)
        obj["confidence"] = round(float(self.confidence), 4)
        obj["score"] = round(float(self.score), 4)
        return obj


def _values(event: EventTuple, field: str) -> List[str]:
    attr = FIELD_MAP[field][0]
    return list(getattr(event, attr))


def _sent(field_cn: str, values: List[str], prefix: str) -> str:
    if not values:
        return f"{prefix}未明确给出{field_cn}。"
    return f"{prefix}{field_cn}为：" + "、".join(values) + "。"


def judge_field(current_event: EventTuple, true_event: EventTuple, field: str) -> FieldNLIResult:
    """Field-wise contradiction detector with an NLI-shaped output.

    The interface mirrors field-wise NLI: entailment / contradiction / neutral /
    unknown. In offline demo mode it uses normalized event-slot alignment. A
    transformer NLI model can replace this function without changing the UI or
    report schema.
    """
    cur = _values(current_event, field)
    tru = _values(true_event, field)
    field_cn = FIELD_MAP[field][1]
    score = list_similarity(cur, tru)

    if not cur and not tru:
        label = "unknown"
        conf = 0.50
        rationale = f"两侧都没有抽取到{field_cn}，不把缺失当作冲突。"
    elif not cur or not tru:
        label = "neutral"
        conf = 0.55
        rationale = f"只有一侧给出{field_cn}，当前证据不足以判定矛盾。"
    elif score >= 0.72:
        label = "entailment"
        conf = max(0.70, score)
        rationale = f"两侧{field_cn}高度一致。"
    elif score < 0.40:
        label = "contradiction"
        conf = 1.0 - score
        rationale = f"两侧{field_cn}明显不同，构成候选冲突字段。"
    else:
        label = "neutral"
        conf = 0.60
        rationale = f"两侧{field_cn}部分相似但不足以判定强矛盾。"

    return FieldNLIResult(
        field=field,
        label=label,
        confidence=max(0.0, min(1.0, conf)),
        current_values=cur,
        true_values=tru,
        score=score,
        hypothesis=_sent(field_cn, cur, "当前 caption 声称"),
        evidence=_sent(field_cn, tru, "图片真实语境显示"),
        rationale=rationale,
    )


def run_field_nli(current_event: EventTuple, true_event: EventTuple) -> Dict[str, Dict[str, object]]:
    return {field: judge_field(current_event, true_event, field).to_dict() for field in FIELD_MAP}
