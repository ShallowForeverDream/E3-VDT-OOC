from __future__ import annotations

from scripts.infer.infer_vdt_cf_attr import UNCERTAIN_TYPE, postprocess_prediction


def _feat(**presence):
    fields = ["entity", "location", "time", "event_type", "relation"]
    out = {
        "image_loaded": 1,
        "clip_enabled": 1,
    }
    # Lower prompt similarity means weaker grounding and is chosen during
    # conservative re-selection.
    defaults = {
        "entity": 0.80,
        "location": 0.30,
        "time": 0.60,
        "event_type": 0.70,
        "relation": 0.75,
    }
    for f in fields:
        out[f"{f}_present"] = int(presence.get(f, 0))
        out[f"clip_prompt_sim_{f}"] = defaults[f]
    return out


def test_absent_entity_field_cannot_output_entity_mismatch():
    mismatch_type, fields, applied, reason = postprocess_prediction(
        mismatch_type="entity mismatch",
        conflict_fields={"entity"},
        vdt_label="OOC",
        feat=_feat(entity=0, location=1, time=1),
    )
    assert applied is True
    assert reason.startswith("field_absent_constraint")
    assert mismatch_type != "entity mismatch"
    assert "entity" not in fields


def test_absent_entity_with_no_reliable_present_field_becomes_uncertain():
    mismatch_type, fields, applied, reason = postprocess_prediction(
        mismatch_type="entity mismatch",
        conflict_fields={"entity"},
        vdt_label="OOC",
        feat={
            "entity_present": 0,
            "location_present": 0,
            "time_present": 0,
            "event_type_present": 0,
            "relation_present": 0,
            "image_loaded": 1,
            "clip_enabled": 1,
        },
    )
    assert applied is True
    assert reason == "field_absent_constraint_no_valid_field"
    assert mismatch_type == UNCERTAIN_TYPE
    assert fields == {"evidence_insufficient"}


def test_non_ooc_gate_for_invalid_field_outputs_benign():
    mismatch_type, fields, applied, reason = postprocess_prediction(
        mismatch_type="entity mismatch",
        conflict_fields={"entity"},
        vdt_label="Non-OOC",
        feat=_feat(entity=0),
    )
    assert applied is True
    assert reason == "vdt_non_ooc_or_benign_gate"
    assert mismatch_type == "benign illustrative image"
    assert fields == set()


def test_single_field_type_conflict_fields_match_primary_field():
    mismatch_type, fields, applied, reason = postprocess_prediction(
        mismatch_type="location mismatch",
        conflict_fields={"entity"},
        vdt_label="OOC",
        feat=_feat(entity=1, location=1),
    )
    assert applied is True
    assert reason == "single_type_primary_field_enforced"
    assert mismatch_type == "location mismatch"
    assert fields == {"location"}
