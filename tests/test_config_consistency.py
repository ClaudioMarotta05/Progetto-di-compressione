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
