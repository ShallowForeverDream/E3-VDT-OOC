from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from e3vdt.attribution.mismatch import build_explanation, infer_mismatch_type
from e3vdt.event.consistency import compute_event_scores, summarize_scores
from e3vdt.event.extractor import extract_event_tuple
from e3vdt.schemas import PredictionResult

VALID_LABELS = {"OOC", "Non-OOC", "Uncertain"}


class E3VDTPipeline:
    """Unified inference pipeline used by CLI and demo.

    Current implementation is a lightweight, explainable heuristic baseline.
    The final VDT/E3-VDT model should keep this API and replace internals.
    """
    def __init__(self, threshold_ooc: float=0.58, model_version: str="e3-vdt-demo-heuristic-v0.2", classification_policy: str="event_sidecar_demo") -> None:
        self.threshold_ooc=threshold_ooc
        self.model_version=model_version
        self.classification_policy=classification_policy

    def _event_heuristic_decision(self, text: str, has_image_context: bool, conflict_fields: List[str], overall: float) -> Tuple[str, float]:
        """Demo-only classifier used when no VDT baseline prediction is provided.

        The research/experiment path is accuracy-preserving: when a VDT baseline
        label is available, final classification must copy that label exactly and
        keep event fields as a sidecar attribution output.
        """
        if not text.strip() or not has_image_context:
            return "Uncertain", 0.35
        if conflict_fields or overall < self.threshold_ooc:
            return "OOC", max(0.55, min(0.98, 1.0-overall+0.25*len(conflict_fields)/5))
        return "Non-OOC", max(0.55, min(0.98, overall))

    def predict(
        self,
        text: str,
        image_path: Optional[str]=None,
        image_context: str="",
        evidence: Optional[List[Dict[str,Any]]]=None,
        baseline_label: Optional[str]=None,
        baseline_score: Optional[float]=None,
        classification_policy: Optional[str]=None,
    ) -> PredictionResult:
        text=text or ""; image_context=image_context or ""; evidence=evidence or []; warnings=[]
        if not text.strip(): warnings.append("文本为空：无法进行可靠检测。")
        if not image_context.strip():
            warnings.append("未提供图像上下文：demo 无法直接理解图片内容，建议填写 image caption/OCR/original context。")
            if image_path: image_context=Path(image_path).stem.replace("_"," ").replace("-"," ")
        text_event=extract_event_tuple(text, source="caption/text")
        image_event=extract_event_tuple(image_context, source="image_context/evidence")
        scores=compute_event_scores(text_event, image_event)
        overall,_=summarize_scores(scores)
        has_image_context=bool(image_context.strip()) and not all(v == 0.5 for v in scores.values())
        mismatch_type, conflict_fields=infer_mismatch_type(scores, has_image_context)
        event_label, event_confidence = self._event_heuristic_decision(text, has_image_context, conflict_fields, overall)

        policy = classification_policy or self.classification_policy
        if baseline_label is not None and baseline_label not in VALID_LABELS:
            warnings.append(f"baseline_label={baseline_label!r} 不在允许集合 {sorted(VALID_LABELS)} 中，已回退到 demo heuristic。")
            baseline_label = None

        if policy in {"baseline_preserving", "sidecar", "accuracy_preserving"} and baseline_label:
            # Hard constraint: do not let attribution/event fields override VDT.
            label = baseline_label
            confidence = float(baseline_score) if baseline_score is not None else event_confidence
            decision_source = "vdt_baseline"
        elif policy in {"baseline_preserving", "sidecar", "accuracy_preserving"}:
            warnings.append("已选择 accuracy-preserving/sidecar 策略，但未提供 VDT baseline_label；当前仅使用 demo heuristic 作为前端演示。")
            label = event_label
            confidence = event_confidence
            decision_source = "event_heuristic_demo_fallback"
        else:
            label = event_label
            confidence = event_confidence
            decision_source = "event_heuristic_demo"

        explanation=build_explanation(label, mismatch_type, conflict_fields, scores)
        return PredictionResult(
            label=label,
            confidence=confidence,
            mismatch_type=mismatch_type,
            conflict_fields=conflict_fields,
            event_scores=scores,
            text_event=text_event,
            image_event=image_event,
            evidence=evidence,
            explanation=explanation,
            model_version=self.model_version,
            classification_policy=policy,
            decision_source=decision_source,
            baseline_label=baseline_label,
            baseline_score=baseline_score,
            warnings=warnings,
        )
    def predict_dict(self, **kwargs: Any) -> Dict[str,Any]:
        return self.predict(**kwargs).to_dict()

def dumps_result(result: PredictionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
