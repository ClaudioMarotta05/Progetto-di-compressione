"""Selezione delle parole da elidere, secondo le condizioni del protocollo."""

import math
import random
from pathlib import Path
from typing import List, Set

from .tokenizer import Token


def select_random_stratified(
    word_indices: List[int],
    target: int,
    seed: int,
    n_blocks: int = 10,
) -> Set[int]:
    """Baseline: selezione uniforme per blocco, senza logica contestuale."""
    if target >= len(word_indices):
        return set(word_indices)
    rng = random.Random(seed)
    selected: Set[int] = set()
    per_block = [0] * n_blocks
    for b in range(n_blocks):
        per_block[b] = (target * (b + 1)) // n_blocks - (target * b) // n_blocks
    block_size = len(word_indices) // n_blocks
    for b in range(n_blocks):
        start = b * block_size
        end = len(word_indices) if b == n_blocks - 1 else (b + 1) * block_size
        block = word_indices[start:end]
        need = per_block[b]
        if need == 0 or not block:
            continue
        if need >= len(block):
            selected.update(block)
            continue
        selected.update(rng.sample(block, need))
    return selected


def select_function_only(
    words: List[Token],
    word_indices: List[int],
    target: int,
    seed: int,
    stopword_path: Path,
) -> Set[int]:
    """Condizione: elide solo function words. Lascia intatte le parole di contenuto.

    Razionale: le function words sono le più prevedibili dal contesto
    (es. "il ___ mangia il ___" -> l'articolo è quasi sempre inferibile).
    Lascia le parole di contenuto come ancore.
    """
    if not stopword_path.exists():
        raise FileNotFoundError(
            f"Lista stopword non trovata: {stopword_path}. "
            f"Esegui experiments/build_external_lists.py."
        )
    stopwords = set(stopword_path.read_text(encoding="utf-8").split())
    candidates = [i for i in word_indices if words[i].text.lower() in stopwords]
    return select_random_stratified(candidates, target, seed)


def select_content_band(
    words: List[Token],
    word_indices: List[int],
    target: int,
    seed: int,
    freq_path: Path,
    min_rank: int,
    max_rank: int,
) -> Set[int]:
    """Condizione: elide parole di contenuto in una fascia di frequenza intermedia.

    Razionale: esclude function words (già economiche per i compressori)
    e nomi propri/tecnicismi (troppo rari per essere inferiti). Elide
    parole che sono costose per un compressore generico ma ancora
    inferibili dal contesto.
    """
    if not freq_path.exists():
        raise FileNotFoundError(
            f"Lista di frequenza non trovata: {freq_path}. "
            f"Esegui experiments/build_external_lists.py."
        )
    rank: dict[str, int] = {}
    for i, line in enumerate(freq_path.read_text(encoding="utf-8").splitlines(), start=1):
        w = line.strip()
        if w:
            rank[w] = i
    candidates = [
        i for i in word_indices
        if min_rank <= rank.get(words[i].text.lower(), 10**9) <= max_rank
    ]
    return select_random_stratified(candidates, target, seed)


def select_anchor_preservation(
    words: List[Token],
    word_indices: List[int],
    target: int,
    seed: int,
    idf_path: Path,
    top_k: int = 3,
    window: int = 50,
) -> Set[int]:
    """Condizione: tieni le ancore contestuali, elidi il resto.

    Razionale: per ogni finestra di 'window' parole, le top_k per IDF
    sono le più informative (danno contesto semantico). Vengono
    preservate come ancore. Si elidono le altre, che il contesto
    residuo rende prevedibili. È l'implementazione della tua intuizione:
    'tenere gatto e pesce, elidere il e mangia'.
    """
    if not idf_path.exists():
        raise FileNotFoundError(
            f"File IDF non trovato: {idf_path}. "
            f"Esegui experiments/build_external_lists.py."
        )
    idf: dict[str, float] = {}
    for line in idf_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            idf[parts[0]] = float(parts[1])

    # Identifica ancore: top_k per IDF in ogni finestra di 'window' parole
    anchors: Set[int] = set()
    for w_start in range(0, len(word_indices), window):
        w_end = min(w_start + window, len(word_indices))
        window_idx = word_indices[w_start:w_end]
        scored = [(idf.get(words[i].text.lower(), 0.0), i) for i in window_idx]
        scored.sort(reverse=True)
        for _, i in scored[:top_k]:
            anchors.add(i)

    # Candidati: tutte le parole non ancora
    candidates = [i for i in word_indices if i not in anchors]
    return select_random_stratified(candidates, target, seed)


def count_candidates(
    condition: dict,
    words: List[Token],
    word_indices: List[int],
    base_dir: Path,
) -> int:
    """Numero di parole eleggibili per la condizione (numeratore del tetto)."""
    method = condition["method"]
    params = condition.get("params", {})
    if method == "random_stratified":
        return len(word_indices)
    if method == "stopword_only":
        path = base_dir / params["stopword_list_ref"]
        stopwords = set(path.read_text(encoding="utf-8").split())
        return sum(1 for i in word_indices if words[i].text.lower() in stopwords)
    if method == "frequency_band":
        path = base_dir / params["frequency_list_ref"]
        band = params["band"]
        rank = {}
        for j, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            w = line.strip()
            if w:
                rank[w] = j
        return sum(
            1 for i in word_indices
            if band["min_rank"] <= rank.get(words[i].text.lower(), 10**9) <= band["max_rank"]
        )
    if method == "tfidf_anchor":
        path = base_dir / params["idf_ref"]
        idf = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                idf[parts[0]] = float(parts[1])
        top_k = params.get("anchor_top_k_per_window", 3)
        window = params.get("window_words", 50)
        anchors = set()
        for w_start in range(0, len(word_indices), window):
            w_end = min(w_start + window, len(word_indices))
            scored = [(idf.get(words[i].text.lower(), 0.0), i) for i in word_indices[w_start:w_end]]
            scored.sort(reverse=True)
            for _, i in scored[:top_k]:
                anchors.add(i)
        return sum(1 for i in word_indices if i not in anchors)
    raise ValueError(f"Metodo sconosciuto: {method}")
