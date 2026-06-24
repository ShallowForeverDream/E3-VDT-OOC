from e3vdt.inference.pipeline import E3VDTPipeline

def test_pipeline_outputs_schema():
    out=E3VDTPipeline().predict_dict(text="A protest erupted in Paris on Monday.", image_context="People gathered in London during an earlier climate demonstration in 2020.")
    assert out["label"] in {"OOC","Non-OOC","Uncertain"}
    assert "mismatch_type" in out and "event_scores" in out
    assert "location" in out["event_scores"]
