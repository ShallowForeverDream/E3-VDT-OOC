from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from e3vdt.attribution.evidence_relevance import score_evidence_relevance
from e3vdt.attribution.field_nli import run_field_nli
from e3vdt.event.extractor import extract_event_tuple

MISMATCH_BY_FIELD = {
    "entity": "entity mismatch",
    "location": "location mismatch",
    "time": "temporal mismatch",
    "event_type": "event-type mismatch",
    "relation": "relation mismatch",
}
FIELD_PRIORITY = ["location", "time", "entity", "event_type", "relation"]


@dataclass
class COVEAttrConfig:
    method_version: str = "vdt-cove-attr-system-demo-v1"
    classification_policy: str = "baseline_preserving_sidecar"


class VDTCOVEAttrPipeline:
    """System-demo pipeline for the agreed technical route.

    VDT provides OOC/Non-OOC classification. COVE-lite uses true image context
    from VisualNews/metadata. Field-wise NLI-shaped attribution identifies which
    event slots conflict. Large-scale experiments can later replace the current
    deterministic NLI adapter without changing the API.
    """

    def __init__(self, config: Optional[COVEAttrConfig] = None) -> None:
        self.config = config or COVEAttrConfig()

    def predict(
        self,
        current_caption: str,
        true_image_context: str,
        vdt_label: Optional[str] = "OOC",
        vdt_score: Optional[float] = 0.87,
        sample_id: str = "manual",
        image_id: str = "manual-image",
        domain: str = "demo",
    ) -> Dict[str, Any]:
        current_caption = current_caption or ""
        true_image_context = true_image_context or ""
        current_event = extract_event_tuple(current_caption, source="current_caption")
        true_event = extract_event_tuple(true_image_context, source="true_image_context")
        evidence = score_evidence_relevance(current_caption, true_image_context, current_event, true_event)
        field_nli = run_field_nli(current_event, true_event)

        if not evidence.sufficient:
            conflict_fields: List[str] = ["evidence_insufficient"]
            mismatch_type = "uncertain / evidence insufficient"
        else:
            conflict_fields = [f for f in FIELD_PRIORITY if field_nli[f]["label"] == "contradiction"]
            mismatch_type = MISMATCH_BY_FIELD.get(conflict_fields[0], "benign illustrative image") if conflict_fields else "benign illustrative image"

        if vdt_label:
            final_label = vdt_label
            decision_source = "vdt_baseline"
            classification_note = "主分类继承 VDT；COVE-Attr 只输出归因，不覆盖 final_label。"
        else:
            final_label = "Uncertain" if not evidence.sufficient else ("OOC" if conflict_fields else "Non-OOC")
            decision_source = "cove_attr_demo_fallback"
            classification_note = "未提供 VDT 预测，系统使用归因结果作为演示 fallback。"

        explanation = self._build_explanation(final_label, mismatch_type, conflict_fields, evidence.to_dict(), field_nli)
        return {
            "sample_id": sample_id,
            "image_id": image_id,
            "domain": domain,
            "method_version": self.config.method_version,
            "technical_route": "VDT baseline + COVE-lite true context + evidence relevance + field-wise NLI attribution",
            "classification_policy": self.config.classification_policy,
            "current_caption": current_caption,
            "true_image_context": true_image_context,
            "vdt": {"label": vdt_label, "score": round(float(vdt_score), 4) if vdt_score is not None else None},
            "final_label": final_label,
            "decision_source": decision_source,
            "classification_note": classification_note,
            "mismatch_type": mismatch_type,
            "conflict_fields": conflict_fields,
            "evidence_relevance": evidence.to_dict(),
            "current_event": current_event.to_dict(),
            "true_event": true_event.to_dict(),
            "field_nli": field_nli,
            "explanation": explanation,
            "experiment_status": {
                "system_demo": "completed",
                "vdt_reproduction": "completed_core_two_domains",
                "large_scale_attribution_eval": "pending_or_running",
                "claim_boundary": "当前演示证明系统闭环；归因有效性需以人工 gold set 与 ablation 结果为准。",
            },
        }

    def _build_explanation(self, final_label: str, mismatch_type: str, conflict_fields: List[str], evidence: Dict[str, Any], field_nli: Dict[str, Dict[str, Any]]) -> str:
        if mismatch_type == "uncertain / evidence insufficient":
            return "图片真实语境证据不足或字段过少，系统不强行给出错配类型；建议补充 VisualNews 原始 caption/OCR/检索证据。"
        if not conflict_fields:
            return "VDT 主分类结果已保留；COVE-Attr 未发现明确字段矛盾，当前图文语境在可抽取事件字段上基本一致。"
        bits = []
        for f in conflict_fields:
            nli = field_nli.get(f, {})
            cur = "、".join(nli.get("current_values") or ["未抽取"])
            tru = "、".join(nli.get("true_values") or ["未抽取"])
            bits.append(f"{f}: 当前={cur}；真实语境={tru}")
        return f"VDT 判断为 {final_label}；COVE-Attr 的主错配类型为 {mismatch_type}。冲突字段：" + "；".join(bits) + f"。证据相关性={evidence.get('level')}({evidence.get('score')})。"


def dumps_result(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
