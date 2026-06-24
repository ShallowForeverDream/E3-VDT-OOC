from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from e3vdt.attribution.mismatch import build_explanation, infer_mismatch_type
from e3vdt.event.consistency import compute_event_scores, summarize_scores
from e3vdt.event.extractor import extract_event_tuple
from e3vdt.schemas import PredictionResult

class E3VDTPipeline:
    """Unified inference pipeline used by CLI and demo.

    Current implementation is a lightweight, explainable heuristic baseline.
    The final VDT/E3-VDT model should keep this API and replace internals.
    """
    def __init__(self, threshold_ooc: float=0.58, model_version: str="e3-vdt-demo-heuristic-v0.1") -> None:
        self.threshold_ooc=threshold_ooc; self.model_version=model_version
    def predict(self, text: str, image_path: Optional[str]=None, image_context: str="", evidence: Optional[List[Dict[str,Any]]]=None) -> PredictionResult:
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
        if not text.strip() or not has_image_context:
            label="Uncertain"; confidence=0.35
        elif conflict_fields or overall < self.threshold_ooc:
            label="OOC"; confidence=max(0.55, min(0.98, 1.0-overall+0.25*len(conflict_fields)/5))
        else:
            label="Non-OOC"; confidence=max(0.55, min(0.98, overall))
        explanation=build_explanation(label, mismatch_type, conflict_fields, scores)
        return PredictionResult(label=label, confidence=confidence, mismatch_type=mismatch_type, conflict_fields=conflict_fields, event_scores=scores, text_event=text_event, image_event=image_event, evidence=evidence, explanation=explanation, model_version=self.model_version, warnings=warnings)
    def predict_dict(self, **kwargs: Any) -> Dict[str,Any]:
        return self.predict(**kwargs).to_dict()

def dumps_result(result: PredictionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
