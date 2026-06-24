from __future__ import annotations
from typing import Dict, List, Tuple
PRIORITY=["location","time","entity","event_type","relation"]
TYPE_BY_FIELD={"entity":"entity mismatch","location":"location mismatch","time":"temporal mismatch","event_type":"event-type mismatch","relation":"relation mismatch"}
def infer_mismatch_type(scores: Dict[str,float], has_image_context: bool) -> Tuple[str,List[str]]:
    if not has_image_context: return "uncertain / evidence insufficient", []
    conflict_fields=[f for f in PRIORITY if scores.get(f,0.5) < 0.40]
    if not conflict_fields:
        if sum(1 for v in scores.values() if 0.40 <= v <= 0.55) >= 3: return "context omission", []
        return "benign illustrative image", []
    return TYPE_BY_FIELD.get(conflict_fields[0], "context omission"), conflict_fields
def build_explanation(label: str, mismatch_type: str, conflict_fields: List[str], scores: Dict[str,float]) -> str:
    score_bits=", ".join(f"{k}={v:.2f}" for k,v in scores.items())
    if label == "OOC":
        fields="、".join(conflict_fields) if conflict_fields else "上下文"
        return f"系统判断为 OOC：主要错配类型为 {mismatch_type}，冲突字段为 {fields}。事件一致性分数：{score_bits}。"
    if label == "Non-OOC":
        return f"系统判断为 Non-OOC：未发现强冲突字段，事件一致性总体较高。事件一致性分数：{score_bits}。"
    return f"系统暂判为 Uncertain：图像侧上下文或外部证据不足，无法可靠判断具体错配类型。当前事件分数：{score_bits}。"
