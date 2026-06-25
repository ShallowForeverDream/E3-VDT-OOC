from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


VALID_LABELS = {"OOC", "Non-OOC", "Uncertain"}
NON_OOC_TYPES = {"", "none", "benign illustrative image"}


class VDTAdapterNotReady(RuntimeError):
    """Backward-compatible exception name.

    The adapter is no longer an empty stub.  This exception is only kept so
    older imports do not break; default inference returns an `Uncertain`
    prediction instead of raising.
    """


@dataclass
class VDTPrediction:
    label: str
    score: float
    ooc_probability: float
    decision_source: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.score

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["score"] = round(float(self.score), 4)
        obj["confidence"] = obj["score"]
        obj["ooc_probability"] = round(float(self.ooc_probability), 4)
        return obj


def _as_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_binary_label(row: Dict[str, Any]) -> int:
    typ = str(row.get("gold_mismatch_type") or "").strip().lower()
    return 0 if typ in NON_OOC_TYPES else 1


class VDTAdapter:
    """Automatic OOC / Non-OOC adapter used by the demo and CLI.

    说明：
    - 如果本机已经跑过 no-true-context 实验，adapter 会用生成的
      `image_caption_features_train/val.csv` 临时训练一个轻量二分类头，
      自动输出 `OOC / Non-OOC`。
    - 如果没有本地特征表，则退回到 CLIP image-caption similarity。
    - 如果没有图片但有 `image_context`，则用事件一致性作为 COVE/oracle
      页面 fallback。

    这不是把官方 VDT checkpoint 强行包装成在线推理；它是一个
    VDT-compatible automatic classifier，用来替代此前网页端手填
    `VDT label / score` 的不可验收空壳。
    """

    _FEATURE_HEAD_CACHE: Dict[Tuple[str, str, bool], Optional[Dict[str, Any]]] = {}

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        project_root: Optional[str | Path] = None,
        feature_dir: Optional[str | Path] = None,
        prefer_feature_head: bool = True,
        clip_model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        no_clip: bool = False,
        clip_center: float = 0.255,
        clip_scale: float = 0.025,
        event_threshold_ooc: float = 0.58,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else ROOT
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.feature_dir = Path(feature_dir).resolve() if feature_dir else None
        self.prefer_feature_head = prefer_feature_head
        self.clip_model = clip_model
        self.device = device or self._guess_device()
        env_no_clip = (
            os.environ.get("VDT_ADAPTER_NO_CLIP", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.environ.get("VDT_CF_ATTR_NO_CLIP", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.no_clip = bool(no_clip or env_no_clip)
        self.clip_center = clip_center
        self.clip_scale = max(clip_scale, 1e-6)
        self.event_threshold_ooc = event_threshold_ooc
        self.init_warnings: List[str] = []

        if self.checkpoint_path and not self.checkpoint_path.exists():
            self.init_warnings.append(
                f"checkpoint_path={self.checkpoint_path} 不存在；当前使用自动 adapter/fallback。"
            )

        self._feature_head: Optional[Dict[str, Any]] = None
        self._feature_head_loaded = not prefer_feature_head

    @staticmethod
    def _guess_device() -> str:
        explicit = os.environ.get("VDT_ADAPTER_DEVICE") or os.environ.get("VDT_CF_ATTR_DEVICE")
        if explicit:
            return explicit
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _candidate_feature_dirs(self) -> List[Path]:
        dirs: List[Path] = []
        env_dir = os.environ.get("VDT_ADAPTER_FEATURE_DIR")
        if env_dir:
            dirs.append(Path(env_dir))
        if self.feature_dir:
            dirs.append(self.feature_dir)
        dirs.extend(
            [
                self.project_root / "outputs" / "no_true_context_attr_5way_plus2000",
                self.project_root / "outputs" / "no_true_context_attr_5way_1000",
                self.project_root / "outputs" / "no_true_context_attr",
            ]
        )
        out: List[Path] = []
        seen = set()
        for d in dirs:
            p = d if d.is_absolute() else self.project_root / d
            key = str(p.resolve())
            if key not in seen:
                out.append(p)
                seen.add(key)
        return out

    def _load_feature_head(self) -> Optional[Dict[str, Any]]:
        cache_key = (str(self.project_root), str(self.feature_dir or ""), bool(self.prefer_feature_head))
        if cache_key in self._FEATURE_HEAD_CACHE:
            return self._FEATURE_HEAD_CACHE[cache_key]

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            from scripts.train.train_no_true_context_attribution_head import feature_names
        except Exception as exc:
            self.init_warnings.append(f"无法加载 sklearn feature-head 后端：{exc}")
            self._FEATURE_HEAD_CACHE[cache_key] = None
            return None

        for d in self._candidate_feature_dirs():
            train = d / "image_caption_features_train.csv"
            val = d / "image_caption_features_val.csv"
            rows: List[Dict[str, Any]] = []
            for p in [train, val]:
                if p.exists():
                    rows.extend(_read_csv(p))
            if len(rows) < 20:
                continue
            names = feature_names(rows, groups=("clip", "field", "vdt"))
            if not names:
                continue
            y = [_normalize_binary_label(r) for r in rows]
            if len(set(y)) < 2:
                continue
            X = np.array([[_as_float(r.get(k)) for k in names] for r in rows], dtype=float)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026),
            )
            model.fit(X, y)
            head = {
                "model": model,
                "feature_names": names,
                "feature_dir": str(d),
                "train_rows": len(rows),
                "backend": "local_no_true_context_feature_logreg",
            }
            self._FEATURE_HEAD_CACHE[cache_key] = head
            return head

        self._FEATURE_HEAD_CACHE[cache_key] = None
        return None

    def _ensure_feature_head(self) -> Optional[Dict[str, Any]]:
        if not self.prefer_feature_head:
            return None
        if not self._feature_head_loaded:
            self._feature_head = self._load_feature_head()
            self._feature_head_loaded = True
        return self._feature_head

    def _build_feature_row(self, image_path: str, caption: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        try:
            from scripts.infer.infer_vdt_cf_attr import build_feature_row

            feat = build_feature_row(
                image_path=image_path,
                caption=caption,
                vdt_score=0.5,
                clip_model=self.clip_model,
                device=self.device,
                no_clip=self.no_clip,
            )
            return feat, warnings
        except Exception as exc:
            warnings.append(f"图像-文本特征构造失败：{exc}")
            return None, warnings

    def _label_from_ooc_probability(self, p_ooc: float, uncertain_band: float = 0.08) -> Tuple[str, float]:
        p_ooc = max(0.0, min(1.0, float(p_ooc)))
        score = max(p_ooc, 1.0 - p_ooc)
        if abs(p_ooc - 0.5) <= uncertain_band:
            return "Uncertain", score
        label = "OOC" if p_ooc > 0.5 else "Non-OOC"
        return label, score

    def _predict_with_feature_head(self, feat: Dict[str, Any]) -> Optional[VDTPrediction]:
        head = self._ensure_feature_head()
        if not head:
            return None
        try:
            import numpy as np

            names: Sequence[str] = head["feature_names"]
            X = np.array([[_as_float(feat.get(k)) for k in names]], dtype=float)
            model = head["model"]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                p_ooc = float(probs[classes.index(1)]) if 1 in classes else float(probs[-1])
            else:
                p_ooc = float(model.predict(X)[0])
            label, score = self._label_from_ooc_probability(p_ooc, uncertain_band=0.04)
            return VDTPrediction(
                label=label,
                score=score,
                ooc_probability=p_ooc,
                decision_source="vdt_adapter_feature_head",
                details={
                    "backend": head.get("backend"),
                    "feature_dir": head.get("feature_dir"),
                    "train_rows": head.get("train_rows"),
                    "clip_enabled": bool(feat.get("clip_enabled")),
                    "image_loaded": bool(feat.get("image_loaded")),
                    "clip_caption_sim": round(_as_float(feat.get("clip_caption_sim")), 4),
                },
                warnings=list(self.init_warnings),
            )
        except Exception as exc:
            self.init_warnings.append(f"feature-head 预测失败，已回退：{exc}")
            return None

    def _predict_with_clip_similarity(self, feat: Dict[str, Any]) -> Optional[VDTPrediction]:
        if not feat.get("image_loaded") or not feat.get("clip_enabled"):
            return None
        sim = _as_float(feat.get("clip_caption_sim"))
        # CLIP 相似度越低，图文越可能 OOC。这里是未校准 fallback，
        # 真正有本地训练输出时会优先走 feature-head。
        p_ooc = 1.0 / (1.0 + math.exp((sim - self.clip_center) / self.clip_scale))
        label, score = self._label_from_ooc_probability(p_ooc, uncertain_band=0.08)
        return VDTPrediction(
            label=label,
            score=score,
            ooc_probability=p_ooc,
            decision_source="vdt_adapter_clip_similarity_fallback",
            details={
                "clip_model": self.clip_model,
                "clip_caption_sim": round(sim, 4),
                "clip_center": self.clip_center,
                "clip_scale": self.clip_scale,
                "image_loaded": bool(feat.get("image_loaded")),
                "clip_enabled": bool(feat.get("clip_enabled")),
                "calibrated": False,
            },
            warnings=list(self.init_warnings)
            + ["未找到本地 feature-head，当前使用 CLIP similarity fallback；该分数仅用于系统演示。"],
        )

    def _predict_with_event_context(self, caption: str, image_context: str) -> VDTPrediction:
        from e3vdt.attribution.mismatch import infer_mismatch_type
        from e3vdt.event.consistency import compute_event_scores, summarize_scores
        from e3vdt.event.extractor import extract_event_tuple

        warnings = list(self.init_warnings)
        caption = caption or ""
        image_context = image_context or ""
        if not caption.strip() or not image_context.strip():
            warnings.append("缺少 caption 或 image_context，无法进行事件一致性判断。")
            return VDTPrediction(
                label="Uncertain",
                score=0.5,
                ooc_probability=0.5,
                decision_source="vdt_adapter_insufficient_input",
                details={},
                warnings=warnings,
            )

        text_event = extract_event_tuple(caption, source="caption")
        image_event = extract_event_tuple(image_context, source="image_context")
        scores = compute_event_scores(text_event, image_event)
        overall, low_fields = summarize_scores(scores)
        has_context = not all(abs(v - 0.5) < 1e-9 for v in scores.values())
        mismatch_type, conflict_fields = infer_mismatch_type(scores, has_context)
        if not has_context:
            label = "Uncertain"
            p_ooc = 0.5
            score = 0.5
        elif conflict_fields or overall < self.event_threshold_ooc:
            label = "OOC"
            p_ooc = max(0.55, min(0.98, 1.0 - overall + 0.12 * len(conflict_fields)))
            score = p_ooc
        else:
            label = "Non-OOC"
            p_ooc = max(0.02, min(0.45, 1.0 - overall))
            score = 1.0 - p_ooc

        return VDTPrediction(
            label=label,
            score=score,
            ooc_probability=p_ooc,
            decision_source="vdt_adapter_event_context_fallback",
            details={
                "mismatch_type_hint": mismatch_type,
                "conflict_fields": conflict_fields or low_fields,
                "event_scores": {k: round(float(v), 4) for k, v in scores.items()},
                "overall_event_consistency": round(float(overall), 4),
            },
            warnings=warnings
            + ["当前没有调用官方 VDT checkpoint；COVE/oracle 页使用事件一致性自动产生 VDT-compatible label。"],
        )

    def predict(
        self,
        image_path: Optional[str] = None,
        caption: str = "",
        text: str = "",
        image_context: str = "",
        strict: bool = False,
        **_: Any,
    ) -> VDTPrediction:
        caption = caption or text or ""
        warnings = list(self.init_warnings)

        if image_path:
            feat, feat_warnings = self._build_feature_row(str(image_path), caption)
            warnings.extend(feat_warnings)
            if feat is not None:
                pred = self._predict_with_feature_head(feat)
                if pred is not None:
                    pred.warnings = list(dict.fromkeys(pred.warnings + warnings))
                    return pred
                pred = self._predict_with_clip_similarity(feat)
                if pred is not None:
                    pred.warnings = list(dict.fromkeys(pred.warnings + warnings))
                    return pred
            if image_context.strip():
                return self._predict_with_event_context(caption, image_context)

        if image_context.strip():
            return self._predict_with_event_context(caption, image_context)

        message = "未提供可用图片或 image_context，VDTAdapter 返回 Uncertain。"
        if strict:
            raise VDTAdapterNotReady(message)
        return VDTPrediction(
            label="Uncertain",
            score=0.5,
            ooc_probability=0.5,
            decision_source="vdt_adapter_insufficient_input",
            details={},
            warnings=list(dict.fromkeys(warnings + [message])),
        )

    def predict_dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.predict(**kwargs).to_dict()
