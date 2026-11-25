"""Controlli sperimentali congelati in config/metrics.yaml: context_density e elided_composition."""
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from scipy.stats import pearsonr, spearmanr

from elision.tokenizer import Token, tokenize
from evaluation.fidelity import extract_words


@dataclass
class ExternalLexicons:
    stopwords: Set[str]
    freq_rank: Dict[str, int]
    oov_rank: int
    idf: Dict[str, float]


def load_external_lexicons(base_dir: Path, elision_cfg: dict) -> ExternalLexicons:
    """Carica le stesse risorse lessicali usate da elision/word_selection.py."""
    conds = {c["name"]: c for c in elision_cfg["gap_generation"]["word_selection"]["conditions"]}

    sw_path = base_dir / conds["function_only"]["params"]["stopword_list_ref"]
    stopwords = set(sw_path.read_text(encoding="utf-8").split())

    freq_path = base_dir / conds["content_band"]["params"]["frequency_list_ref"]
    freq_rank: Dict[str, int] = {}
    for idx, line in enumerate(freq_path.read_text(encoding="utf-8").splitlines(), start=1):
        w = line.strip()
        if w:
            freq_rank[w] = idx
    oov_rank = len(freq_rank) + 1

    idf_path = base_dir / conds["anchor_preservation"]["params"]["idf_ref"]
    idf: Dict[str, float] = {}
    for line in idf_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            idf[parts[0]] = float(parts[1])

    return ExternalLexicons(stopwords=stopwords, freq_rank=freq_rank, oov_rank=oov_rank, idf=idf)


def identify_anchor_indices(
    tokens: List[Token],
    word_indices: List[int],
    idf: Dict[str, float],
    top_k: int = 3,
    window: int = 50,
) -> Set[int]:
    """Identifica le ancore con la stessa logica esatta di select_anchor_preservation()."""
    anchors: Set[int] = set()
    for w_start in range(0, len(word_indices), window):
        w_end = min(w_start + window, len(word_indices))
        window_idx = word_indices[w_start:w_end]
        scored = [(idf.get(tokens[i].text.lower(), 0.0), i) for i in window_idx]
        scored.sort(reverse=True)
        for _, i in scored[:top_k]:
            anchors.add(i)
    return anchors


def compute_context_density_per_gap(
    original_text: str,
    manifest: List[dict],
    lexicons: ExternalLexicons,
    window_words: int = 50,
    anchor_top_k: int = 3,
    anchor_window: int = 50,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Anchor words residue in ±window_words attorno al gap / parole residue nella finestra."""
    tokens = tokenize(original_text)
    word_indices = [i for i, t in enumerate(tokens) if t.is_word]
    anchors = identify_anchor_indices(
        tokens, word_indices, lexicons.idf, top_k=anchor_top_k, window=anchor_window
    )

    elided_token_indices: Set[int] = set()
    for m in manifest:
        for idx in range(int(m["orig_token_start"]), int(m["orig_token_end"])):
            elided_token_indices.add(idx)

    surviving_word_indices = [i for i in word_indices if i not in elided_token_indices]

    per_gap_density: Dict[int, float] = {}
    for m in sorted(manifest, key=lambda x: int(x["gap_id"])):
        gid = int(m["gap_id"])
        start_tok = int(m["orig_token_start"])
        pos = bisect_left(surviving_word_indices, start_tok)
        left = max(0, pos - window_words)
        right = min(len(surviving_word_indices), pos + window_words)
        win = surviving_word_indices[left:right]
        if not win:
            per_gap_density[gid] = 0.0
        else:
            n_anchors = sum(1 for idx in win if idx in anchors)
            per_gap_density[gid] = n_anchors / len(win)

    vals = list(per_gap_density.values())
    stats = {
        "mean_context_density": (sum(vals) / len(vals)) if vals else 0.0,
        "min_context_density": min(vals) if vals else 0.0,
        "max_context_density": max(vals) if vals else 0.0,
        "window_words": window_words,
    }
    return per_gap_density, stats


def compute_elided_composition(oracle: List[dict], lexicons: ExternalLexicons) -> Dict[str, float]:
    """Calcola content_ratio, function_word_ratio e mean_zipf_rank sulle parole rimosse."""
    removed_words: List[str] = []
    for o in oracle:
        removed_words.extend(extract_words(o.get("removed_text", "")))

    total = len(removed_words)
    if total == 0:
        return {
            "n_removed_words": 0,
            "content_ratio": 0.0,
            "function_word_ratio": 0.0,
            "mean_zipf_rank": 0.0,
            "oov_ratio": 0.0,
        }

    n_func = 0
    ranks: List[int] = []
    n_oov = 0
    for w in removed_words:
        wl = w.lower()
        if wl in lexicons.stopwords:
            n_func += 1
        if wl in lexicons.freq_rank:
            ranks.append(lexicons.freq_rank[wl])
        else:
            ranks.append(lexicons.oov_rank)
            n_oov += 1

    n_content = total - n_func
    return {
        "n_removed_words": total,
        "content_ratio": n_content / total,
        "function_word_ratio": n_func / total,
        "mean_zipf_rank": sum(ranks) / total,
        "oov_ratio": n_oov / total,
    }


def correlate_context_with_accuracy(
    per_gap_density: Dict[int, float],
    per_gap_errors: List[Dict[str, float]],
) -> Dict[str, Optional[float]]:
    """Correlazione (Spearman e Pearson) tra context_density e accuratezza per gap."""
    xs: List[float] = []
    ys: List[float] = []
    for g in per_gap_errors:
        gid = int(g["gap_id"])
        if gid in per_gap_density:
            xs.append(per_gap_density[gid])
            ys.append(float(g["gap_accuracy"]))

    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {
            "n_points": len(xs),
            "spearman_r": None,
            "spearman_p": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    sr, sp = spearmanr(xs, ys)
    pr, pp = pearsonr(xs, ys)
    return {
        "n_points": len(xs),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
    }
