import yaml
from pathlib import Path

CONFIG = Path("config")

def load(name):
    return yaml.safe_load((CONFIG / name).read_text())

def test_strategies_defined():
    elision = load("elision.yaml")
    assert set(elision["strategies"].keys()) == {"A", "B"}

def test_ratios_frozen():
    elision = load("elision.yaml")
    assert elision["ratios"]["values"] == [0.10, 0.20, 0.30, 0.40, 0.50]

def test_metrics_streams_match_compression():
    metrics = load("metrics.yaml")
    assert metrics["compression"]["streams"] == [
        "skeleton", "manifest_or_placeholder", "corrections"
    ]

def test_precision_at_k_frozen():
    metrics = load("metrics.yaml")
    assert metrics["fidelity"]["precision_at_k"]["k_values"] == [1, 3, 5, 10]

def test_seed_derivation_consistent():
    base = load("base.yaml")
    elision = load("elision.yaml")
    assert elision["seed"]["derivation"] == "sha256"
    assert elision["seed"]["global_seed_ref"].startswith("config/base.yaml")

def test_base_has_paths_scales_splits():
    base = load("base.yaml")
    assert "paths" in base and "data_dev" in base["paths"]
    assert set(base["scales"].keys()) == {"small", "medium", "large"}
    assert base["splits"]["holdout_fraction"] == 0.20

def test_tokenizer_placeholder_atomic():
    tok = load("tokenizer.yaml")
    assert tok["tokenization_mode"] == "finditer"
    assert tok["placeholder"]["atomic_non_word"] is True

def test_elision_has_four_word_selection_conditions():
    elision = load("elision.yaml")
    names = {c["name"] for c in elision["gap_generation"]["word_selection"]["conditions"]}
    assert names == {"random_stratified", "function_only", "content_band", "anchor_preservation"}

def test_elision_rng_order_frozen():
    elision = load("elision.yaml")
    assert elision["gap_generation"]["rng_call_order"] == ["word", "L", "offset"]

def test_metrics_has_controls():
    metrics = load("metrics.yaml")
    assert "context_density" in metrics["controls"]
    assert "elided_composition" in metrics["controls"]

def test_splits_has_n_docs_per_scale():
    base = load("base.yaml")
    assert base["splits"]["n_docs_per_scale"] == 10













