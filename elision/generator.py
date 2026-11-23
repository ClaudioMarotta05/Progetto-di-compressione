"""Generatore deterministico di bucatura."""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import yaml

from .seed import seed_doc
from .tokenizer import tokenize, tokenize_with_placeholders, Token
from .word_selection import (
    select_random_stratified,
    select_function_only,
    select_content_band,
    select_anchor_preservation,
    count_candidates,
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select_words(
    condition: dict,
    words: List[Token],
    word_indices: List[int],
    target: int,
    seed: int,
    base_dir: Path,
) -> set[int]:
    method = condition["method"]
    params = condition.get("params", {})
    if method == "random_stratified":
        return select_random_stratified(word_indices, target, seed, params.get("n_blocks", 10))
    if method == "stopword_only":
        return select_function_only(
            words, word_indices, target, seed,
            base_dir / params["stopword_list_ref"],
        )
    if method == "frequency_band":
        band = params["band"]
        return select_content_band(
            words, word_indices, target, seed,
            base_dir / params["frequency_list_ref"],
            band["min_rank"], band["max_rank"],
        )
    if method == "tfidf_anchor":
        return select_anchor_preservation(
            words, word_indices, target, seed,
            base_dir / params["idf_ref"],
            params.get("anchor_top_k_per_window", 3),
            params.get("window_words", 50),
        )
    raise ValueError(f"Metodo di selezione sconosciuto: {method}")


def generate(
    text: str,
    doc_id: str,
    strategy: str,
    ratio: float,
    condition: dict,
    config: dict,
    base_dir: Path,
) -> Tuple[str, List[dict], List[dict], dict]:
    global_seed = config["reproducibility"]["global_seed"]
    sd = seed_doc(global_seed, doc_id)

    tokens = tokenize_with_placeholders(text) if strategy == "B" else tokenize(text)
    word_indices = [i for i, t in enumerate(tokens) if t.is_word]
    W = len(word_indices)
    target = int(ratio * W + 0.5)

    n_candidates = count_candidates(condition, tokens, word_indices, base_dir)

    selected = select_words(condition, tokens, word_indices, target, sd, base_dir)

    word_pos_of_token = {tok_idx: pos for pos, tok_idx in enumerate(word_indices)}
    selected_word_positions = sorted(
        word_pos_of_token[i] for i in selected if i in word_pos_of_token
    )
    word_gaps: List[Tuple[int, int]] = []
    if selected_word_positions:
        run_start = selected_word_positions[0]
        run_end = run_start
        for p in selected_word_positions[1:]:
            if p == run_end + 1:
                run_end = p
            else:
                word_gaps.append((run_start, run_end))
                run_start = p
                run_end = p
        word_gaps.append((run_start, run_end))

    gaps = []
    for gap_id, (ws, we) in enumerate(word_gaps):
        start_token = word_indices[ws]
        end_token = word_indices[we]
        if start_token > 0 and not tokens[start_token - 1].is_word:
            start_token -= 1
        removed_text = "".join(tokens[j].text for j in range(start_token, end_token + 1))
        n_removed = we - ws + 1
        gaps.append({
            "doc_id": doc_id,
            "gap_id": gap_id,
            "orig_token_start": start_token,
            "orig_token_end": end_token,
            "n_removed": n_removed,
            "strategy": strategy,
            "ratio_target": ratio,
            "word_selection_condition": condition["name"],
            "removed_text": removed_text,
        })

    out_tokens: List[Token] = []
    manifest: List[dict] = []
    effective_ratio = len(selected) / W if W else 0.0
    gap_idx = 0
    i = 0
    while i < len(tokens):
        if gap_idx < len(gaps) and i == gaps[gap_idx]["orig_token_start"]:
            g = gaps[gap_idx]
            g["skeleton_pos"] = len(out_tokens)
            if strategy == "B":
                out_tokens.append(Token(f"[{g['n_removed']}]", -1, -1, False))
            i = g["orig_token_end"] + 1
            m = {k: v for k, v in g.items() if k != "removed_text"}
            m["effective_ratio"] = effective_ratio
            manifest.append(m)
            gap_idx += 1
        else:
            out_tokens.append(tokens[i])
            i += 1

    skeleton = "".join(t.text for t in out_tokens)

    oracle = [
        {"doc_id": doc_id, "gap_id": g["gap_id"], "removed_text": g["removed_text"]}
        for g in gaps
    ]

    meta = {
        "doc_id": doc_id,
        "strategy": strategy,
        "ratio": ratio,
        "condition": condition["name"],
        "seed_doc": sd,
        "n_words_total": W,
        "n_words_removed": len(selected),
        "n_gaps": len(gaps),
        "effective_ratio": effective_ratio,
        "n_candidates": n_candidates,
        "candidate_ratio": n_candidates / W if W else 0.0,
        "target_reachable": target <= n_candidates,
        "skeleton_sha256": sha256(skeleton),
        "manifest_sha256": sha256(json.dumps(manifest, sort_keys=True)),
        "oracle_sha256": sha256(json.dumps(oracle, sort_keys=True)),
    }

    return skeleton, manifest, oracle, meta
