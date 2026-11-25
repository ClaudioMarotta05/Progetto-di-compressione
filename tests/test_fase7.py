"""Test unitari Fase 7: fedeltà (BLEU, ROUGE-L, Precision@k), errori LCS e controlli."""
from pathlib import Path
import yaml

from elision.generator import generate, load_config
from evaluation.controls import (
    compute_context_density_per_gap,
    compute_elided_composition,
    load_external_lexicons,
)
from evaluation.errors import evaluate_errors, lcs_length
from evaluation.fidelity import compute_bleu, compute_precision_at_k, compute_rouge_l

BASE = Path(".")


def test_lcs_and_error_rates():
    assert lcs_length(["il", "gatto", "nero"], ["il", "cane", "nero"]) == 2
    oracle = [
        {"gap_id": 0, "removed_text": " il gatto"},
        {"gap_id": 1, "removed_text": " pesce"},
    ]
    preds = [
        {"gap_id": 0, "inferred": ["il", "cane", "gatto"]},
        {"gap_id": 1, "inferred": []},
    ]
    summary, per_gap = evaluate_errors(preds, oracle)
    assert summary["counts"]["correct"] == 2
    assert summary["counts"]["hallucinated"] == 1
    assert summary["counts"]["omitted"] == 1
    assert summary["counts"]["wrong_length_gaps"] == 2
    assert summary["counts"]["overinserted"] == 1
    assert len(per_gap) == 2


def test_precision_at_k_edge_cases():
    oracle = [
        {"gap_id": 0, "removed_text": " nel"},
        {"gap_id": 1, "removed_text": " bosco"},
        {"gap_id": 2, "removed_text": " verde"},
    ]
    preds = [
        {"gap_id": 0, "inferred": ["nel"]},
        {"gap_id": 1, "inferred": ["mare"]},
        {"gap_id": 2, "inferred": ["verde"]},
    ]
    pk = compute_precision_at_k(preds, oracle, k_values=[1, 3, 10])
    assert pk["precision_at_1"] == 1.0
    assert abs(pk["precision_at_3"] - (2 / 3)) < 1e-9
    assert abs(pk["precision_at_10"] - (2 / 3)) < 1e-9

    pk_empty = compute_precision_at_k([], [], k_values=[1, 5])
    assert pk_empty["precision_at_1"] is None
    assert pk_empty["precision_at_5"] is None


def test_bleu_and_rouge_identity():
    text = "Cariacica è un comune del Brasile nello Stato dell'Espírito Santo."
    b = compute_bleu([text], [text])
    r = compute_rouge_l(text, text)
    assert b["bleu"] > 99.9
    assert abs(r["rouge_l_f1"] - 1.0) < 1e-9


def test_controls_density_and_composition():
    base_cfg = load_config(BASE / "config/base.yaml")
    elision_cfg = yaml.safe_load((BASE / "config/elision.yaml").read_text(encoding="utf-8"))
    lexicons = load_external_lexicons(BASE, elision_cfg)
    cond = next(
        c for c in elision_cfg["gap_generation"]["word_selection"]["conditions"]
        if c["name"] == "function_only"
    )
    sample = "Il gatto mangia il pesce sul tavolo della cucina nel pomeriggio. " * 15
    _, manifest, oracle, _ = generate(sample, "test_doc", "A", 0.20, cond, base_cfg, BASE)

    dens_map, dens_stats = compute_context_density_per_gap(sample, manifest, lexicons)
    assert len(dens_map) == len(manifest)
    assert 0.0 <= dens_stats["mean_context_density"] <= 1.0

    comp = compute_elided_composition(oracle, lexicons)
    assert comp["n_removed_words"] > 0
    assert comp["function_word_ratio"] > 0.9
    assert abs(comp["function_word_ratio"] + comp["content_ratio"] - 1.0) < 1e-9
