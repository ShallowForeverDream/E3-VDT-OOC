from e3vdt.inference.pipeline import E3VDTPipeline


def test_baseline_preserving_never_overrides_vdt_label_on_conflict():
    pipe = E3VDTPipeline(classification_policy="baseline_preserving")
    out = pipe.predict_dict(
        text="A large protest erupted in Paris on Monday after a new climate policy.",
        image_context="People gathered in London during a climate demonstration on Monday.",
        baseline_label="Non-OOC",
        baseline_score=0.91,
    )

    assert out["label"] == "Non-OOC"
    assert out["baseline_label"] == "Non-OOC"
    assert out["decision_source"] == "vdt_baseline"
    assert out["classification_policy"] == "baseline_preserving"
    assert out["mismatch_type"] == "location mismatch"
    assert "location" in out["conflict_fields"]


def test_baseline_preserving_keeps_ooc_label_on_non_conflict():
    pipe = E3VDTPipeline(classification_policy="baseline_preserving")
    out = pipe.predict_dict(
        text="A flood caused evacuations in Shanghai in 2024.",
        image_context="A flood caused evacuations in Shanghai in 2024.",
        baseline_label="OOC",
        baseline_score=0.88,
    )

    assert out["label"] == "OOC"
    assert out["decision_source"] == "vdt_baseline"
    assert out["mismatch_type"] == "benign illustrative image"
