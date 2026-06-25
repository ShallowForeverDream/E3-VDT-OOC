from e3vdt.inference.cove_attr_pipeline import VDTCOVEAttrPipeline


def test_cove_attr_location_time_conflict_preserves_vdt_label():
    obj = VDTCOVEAttrPipeline().predict(
        current_caption="Protesters marched in Paris on Monday.",
        true_image_context="Demonstrators gathered in London in 2019 during a climate protest.",
        vdt_label="OOC",
        vdt_score=0.91,
    )
    assert obj["final_label"] == "OOC"
    assert obj["decision_source"] == "vdt_baseline"
    assert obj["mismatch_type"] == "location mismatch"
    assert "location" in obj["conflict_fields"]
    assert "time" in obj["conflict_fields"]


def test_cove_attr_evidence_insufficient_gate():
    obj = VDTCOVEAttrPipeline().predict(
        current_caption="A fire broke out in New York in 2024.",
        true_image_context="",
        vdt_label="OOC",
        vdt_score=0.73,
    )
    assert obj["final_label"] == "OOC"
    assert obj["mismatch_type"] == "uncertain / evidence insufficient"
    assert obj["conflict_fields"] == ["evidence_insufficient"]
    assert obj["evidence_relevance"]["sufficient"] is False
