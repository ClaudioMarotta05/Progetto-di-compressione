"""Metriche di fedeltà congelate in config/metrics.yaml (BLEU, ROUGE-L, Precision@k)."""
from typing import Any, Dict, List, Optional, Sequence, Union

import sacrebleu
from rouge_score import rouge_scorer

from elision.tokenizer import tokenize


def extract_words(text_or_tokens: Union[str, Sequence[str]]) -> List[str]:
    """Estrae i token-parola in modo coerente con elision/tokenizer.py (case-sensitive)."""
    if text_or_tokens is None:
        return []
    if isinstance(text_or_tokens, str):
        raw = text_or_tokens
    else:
        raw = " ".join(str(x) for x in text_or_tokens if x is not None)
    return [t.text for t in tokenize(raw) if t.is_word]


def _extract_pred_value(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    for key in (
        "inferred",
        "predicted",
        "inferred_words",
        "predicted_words",
        "inferred_text",
        "predicted_text",
        "filled_text",
        "text",
    ):
        if key in entry and entry[key] is not None:
            return entry[key]
    return []


def build_pred_map(predictions: Any) -> Dict[int, Any]:
    """Normalizza corrections/predictions in una mappa {gap_id: valore_predetto}."""
    out: Dict[int, Any] = {}
    if isinstance(predictions, dict):
        for k, v in predictions.items():
            out[int(k)] = _extract_pred_value(v)
    elif isinstance(predictions, list):
        for p in predictions:
            if isinstance(p, dict) and "gap_id" in p:
                out[int(p["gap_id"])] = _extract_pred_value(p)
    return out


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """sacrebleu corpus-level: tokenizer=13a, case_sensitive=True, smooth=exp, effective_order=True."""
    res = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        smooth_method="exp",
        tokenize="13a",
        lowercase=False,
        use_effective_order=True,
    )
    return {
        "bleu": float(res.score),
        "bp": float(res.bp),
        "sys_len": int(res.sys_len),
        "ref_len": int(res.ref_len),
    }


def compute_rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    """ROUGE-L document-level: F1, precision, recall, use_stemmer=False."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference, hypothesis)["rougeL"]
    return {
        "rouge_l_f1": float(score.fmeasure),
        "rouge_l_precision": float(score.precision),
        "rouge_l_recall": float(score.recall),
    }


def compute_precision_at_k(
    predictions: Any,
    oracle: List[dict],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Optional[float]]:
    """Precision@k sui primi k gap ordinati per gap_id (exact match token-level case-sensitive)."""
    pred_map = build_pred_map(predictions)
    ordered_oracle = sorted(oracle, key=lambda o: int(o["gap_id"]))
    n_gaps = len(ordered_oracle)

    if n_gaps == 0:
        return {f"precision_at_{k}": None for k in k_values}

    matches: List[bool] = []
    for o in ordered_oracle:
        gid = int(o["gap_id"])
        oracle_words = extract_words(o.get("removed_text", ""))
        pred_words = extract_words(pred_map.get(gid, []))
        matches.append(pred_words == oracle_words)

    out: Dict[str, Optional[float]] = {}
    for k in k_values:
        k_eff = min(k, n_gaps)
        out[f"precision_at_{k}"] = sum(matches[:k_eff]) / k_eff
    return out
