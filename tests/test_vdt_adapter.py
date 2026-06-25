from pathlib import Path

from e3vdt.inference.vdt_adapter import VDTAdapter
from scripts.infer.infer_vdt_cf_attr import predict as predict_vdt_cf_attr


ROOT = Path(__file__).resolve().parents[1]


def test_vdt_adapter_event_context_fallback_is_not_stub():
    pred = VDTAdapter(prefer_feature_head=False, no_clip=True).predict(
        caption="Protesters marched in Paris on Monday.",
        image_context="Demonstrators gathered in London in 2019 during a climate protest.",
    )
    assert pred.label == "OOC"
    assert pred.decision_source == "vdt_adapter_event_context_fallback"
    assert pred.ooc_probability > 0.5


def test_vdt_adapter_insufficient_input_returns_uncertain_instead_of_raising():
    pred = VDTAdapter(prefer_feature_head=False, no_clip=True).predict(caption="")
    assert pred.label == "Uncertain"
    assert pred.decision_source == "vdt_adapter_insufficient_input"


def test_no_true_context_infer_can_auto_call_vdt_adapter_without_clip():
    obj = predict_vdt_cf_attr(
        image_path=str(ROOT / "examples" / "demo_images" / "flood_shanghai_2024.png"),
        caption="A flood caused evacuations in Shanghai in 2024.",
        vdt_label="auto",
        model_path=str(ROOT / "missing_attr_head.pkl"),
        device="cpu",
        no_clip=True,
    )
    assert obj["auto_vdt"] is not None
    assert obj["auto_vdt"]["label"] in {"OOC", "Non-OOC", "Uncertain"}
    assert obj["vdt_label"] == obj["auto_vdt"]["label"]


def test_uncertain_vdt_label_gates_off_attribution():
    obj = predict_vdt_cf_attr(
        image_path=str(ROOT / "examples" / "demo_images" / "flood_shanghai_2024.png"),
        caption="A flood caused evacuations in Shanghai in 2024.",
        vdt_label="Uncertain",
        vdt_score=0.5,
        model_path=str(ROOT / "missing_attr_head.pkl"),
        device="cpu",
        no_clip=True,
    )
    assert obj["decision_source"] == "vdt_uncertain_gate"
    assert obj["mismatch_type"] == "uncertain / insufficient visual evidence"
    assert obj["conflict_fields"] == ["evidence_insufficient"]
