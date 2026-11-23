import json
from pathlib import Path

from elision.seed import seed_doc
from elision.tokenizer import tokenize, tokenize_with_placeholders
from elision.generator import generate, load_config


def test_seed_stable():
    assert seed_doc(42, "doc1") == seed_doc(42, "doc1")
    assert seed_doc(42, "doc1") != seed_doc(42, "doc2")
    assert seed_doc(43, "doc1") != seed_doc(42, "doc1")


def test_tokenize_preserves_text():
    text = "Il gatto mangia il pesce."
    tokens = tokenize(text)
    assert "".join(t.text for t in tokens) == text


def test_placeholder_atomic():
    tokens = tokenize_with_placeholders("Il [3] mangia.")
    placeholders = [t for t in tokens if t.text == "[3]"]
    assert len(placeholders) == 1
    assert placeholders[0].is_word is False


def test_generate_deterministic():
    config = load_config(Path("config/base.yaml"))
    elision_cfg = load_config(Path("config/elision.yaml"))
    condition = elision_cfg["gap_generation"]["word_selection"]["conditions"][0]
    text = "Il gatto mangia il pesce. " * 100
    a = generate(text, "doc1", "A", 0.30, condition, config, Path("."))
    b = generate(text, "doc1", "A", 0.30, condition, config, Path("."))
    assert a[3]["skeleton_sha256"] == b[3]["skeleton_sha256"]


def test_oracle_not_in_manifest():
    config = load_config(Path("config/base.yaml"))
    elision_cfg = load_config(Path("config/elision.yaml"))
    condition = elision_cfg["gap_generation"]["word_selection"]["conditions"][0]
    text = "Il gatto mangia il pesce."
    _, manifest, oracle, _ = generate(text, "doc1", "A", 0.50, condition, config, Path("."))
    manifest_str = json.dumps(manifest)
    for o in oracle:
        assert "removed_text" in o
        assert o["removed_text"] not in manifest_str
