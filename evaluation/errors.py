"""Classificazione deterministica degli errori via Longest Common Subsequence (LCS)."""
from typing import Any, Dict, List, Tuple

from evaluation.fidelity import build_pred_map, extract_words


def lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Lunghezza della sottosequenza comune più lunga (allineamento deterministico case-sensitive)."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = seq_a[i - 1]
        for j in range(1, n + 1):
            if ai == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[n]


def evaluate_errors(
    predictions: Any,
    oracle: List[dict],
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """Calcola conteggi e tassi di errore definiti in config/metrics.yaml + dettaglio per gap."""
    pred_map = build_pred_map(predictions)
    ordered_oracle = sorted(oracle, key=lambda o: int(o["gap_id"]))

    total_oracle_tokens = 0
    total_pred_tokens = 0
    total_correct = 0
    total_hallucinated = 0
    total_omitted = 0
    total_wrong_length_gaps = 0
    total_overinserted = 0
    rel_len_errors: List[float] = []
    exact_gap_hits = 0

    per_gap: List[Dict[str, float]] = []

    for o in ordered_oracle:
        gid = int(o["gap_id"])
        o_words = extract_words(o.get("removed_text", ""))
        p_words = extract_words(pred_map.get(gid, []))

        n_o = len(o_words)
        n_p = len(p_words)
        correct = lcs_length(p_words, o_words)
        hallucinated = n_p - correct
        omitted = n_o - correct
        wrong_length = 1 if abs(n_p - n_o) > 0 else 0
        overinserted = max(0, n_p - n_o)
        rel_len_err = abs(n_p - n_o) / n_o if n_o > 0 else float(n_p)
        gap_acc = correct / n_o if n_o > 0 else (1.0 if n_p == 0 else 0.0)
        is_exact = 1 if p_words == o_words else 0

        total_oracle_tokens += n_o
        total_pred_tokens += n_p
        total_correct += correct
        total_hallucinated += hallucinated
        total_omitted += omitted
        total_wrong_length_gaps += wrong_length
        total_overinserted += overinserted
        rel_len_errors.append(rel_len_err)
        exact_gap_hits += is_exact

        per_gap.append(
            {
                "gap_id": gid,
                "n_oracle_tokens": n_o,
                "n_pred_tokens": n_p,
                "correct": correct,
                "hallucinated": hallucinated,
                "omitted": omitted,
                "wrong_length": wrong_length,
                "overinserted": overinserted,
                "gap_accuracy": gap_acc,
                "exact_match": is_exact,
            }
        )

    n_gaps = len(ordered_oracle)
    summary = {
        "n_gaps": n_gaps,
        "total_oracle_tokens": total_oracle_tokens,
        "total_pred_tokens": total_pred_tokens,
        "counts": {
            "correct": total_correct,
            "hallucinated": total_hallucinated,
            "omitted": total_omitted,
            "wrong_length_gaps": total_wrong_length_gaps,
            "overinserted": total_overinserted,
            "exact_gap_hits": exact_gap_hits,
        },
        "rates": {
            "accuracy": (total_correct / total_oracle_tokens) if total_oracle_tokens else 0.0,
            "hallucination_rate": (total_hallucinated / total_pred_tokens) if total_pred_tokens else 0.0,
            "omission_rate": (total_omitted / total_oracle_tokens) if total_oracle_tokens else 0.0,
            "length_error": (sum(rel_len_errors) / n_gaps) if n_gaps else 0.0,
            "overinsertion_rate": (total_overinserted / total_oracle_tokens) if total_oracle_tokens else 0.0,
            "wrong_length_gap_rate": (total_wrong_length_gaps / n_gaps) if n_gaps else 0.0,
            "exact_gap_accuracy": (exact_gap_hits / n_gaps) if n_gaps else 0.0,
        },
    }
    return summary, per_gap
