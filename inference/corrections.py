"""Allineamento deterministico su token tra scheletro e output del modello (A e B)."""
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from elision.tokenizer import tokenize, tokenize_with_placeholders


def _detect_token_index_mode(
    strategy: str, skeleton: str, skel_tokens: list, manifest: list[dict]
) -> bool:
    """Rileva automaticamente se skeleton_pos è un indice di token (True) o di caratteri (False)."""
    if not manifest:
        return True

    if strategy == "B":
        return all(
            0 <= g["skeleton_pos"] < len(skel_tokens)
            and skel_tokens[g["skeleton_pos"]].text == f"[{g['n_removed']}]"
            for g in manifest
        )

    removed_before = 0
    for g in manifest:
        if g["skeleton_pos"] != g["orig_token_start"] - removed_before:
            return False
        removed_before += g["orig_token_end"] - g["orig_token_start"] + 1
    return True


def _gap_word_positions(
    strategy: str, skeleton: str, skel_tokens: list, manifest: list[dict]
) -> list[int]:
    """Converte skeleton_pos di ogni gap nella posizione (tra parole) dentro skel_words."""
    is_tok_idx = _detect_token_index_mode(strategy, skeleton, skel_tokens, manifest)

    words_before_tok = [0] * (len(skel_tokens) + 1)
    for idx, t in enumerate(skel_tokens):
        words_before_tok[idx + 1] = words_before_tok[idx] + (1 if t.is_word else 0)

    positions: list[int] = []
    for g in manifest:
        pos = g["skeleton_pos"]
        if is_tok_idx:
            clamped = max(0, min(pos, len(skel_tokens)))
            positions.append(words_before_tok[clamped])
        else:
            positions.append(sum(1 for t in skel_tokens if t.is_word and t.start < pos))
    return positions


def compute_corrections(
    strategy: str,
    skeleton: str,
    output: str,
    manifest: list[dict],
) -> list[dict]:
    """Estrae le parole inferite per ciascun gap allineando skel_words -> out_words."""
    if strategy not in ("A", "B"):
        raise ValueError(f"Strategia sconosciuta: {strategy!r}")

    skel_tokens = (
        tokenize_with_placeholders(skeleton) if strategy == "B" else tokenize(skeleton)
    )
    out_tokens = tokenize_with_placeholders(output)

    skel_words = [t.text for t in skel_tokens if t.is_word]
    out_words = [t.text for t in out_tokens if t.is_word]

    gap_wpos = _gap_word_positions(strategy, skeleton, skel_tokens, manifest)
    sm = SequenceMatcher(None, skel_words, out_words, autojunk=False)
    opcodes = sm.get_opcodes()

    insert_at = {
        i1: idx for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes) if tag == "insert"
    }
    replaces = [
        (i1, i2, idx)
        for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes)
        if tag == "replace"
    ]

    # Assegna ogni gap all'opcode corrispondente
    op_to_gaps: dict[int, list[int]] = {}
    r_ptr = 0
    for g_idx, w_pos in enumerate(gap_wpos):
        if w_pos in insert_at:
            op_idx = insert_at[w_pos]
            op_to_gaps.setdefault(op_idx, []).append(g_idx)
            continue

        while r_ptr < len(replaces) and replaces[r_ptr][1] < w_pos:
            r_ptr += 1
        if r_ptr < len(replaces) and replaces[r_ptr][0] <= w_pos <= replaces[r_ptr][1]:
            op_idx = replaces[r_ptr][2]
            op_to_gaps.setdefault(op_idx, []).append(g_idx)

    inferred_by_gap: list[list[str]] = [[] for _ in manifest]

    for op_idx, g_indices in op_to_gaps.items():
        tag, i1, i2, j1, j2 = opcodes[op_idx]
        if tag == "insert":
            new_words = out_words[j1:j2]
        else:
            # In un blocco replace, esclude le parole dello scheletro rimaste invariate
            skel_counts = Counter(skel_words[i1:i2])
            new_words = []
            for w in out_words[j1:j2]:
                if skel_counts[w] > 0:
                    skel_counts[w] -= 1
                else:
                    new_words.append(w)

        if len(g_indices) == 1:
            inferred_by_gap[g_indices[0]] = new_words
        else:
            cursor = 0
            for k, g_idx in enumerate(g_indices):
                n_exp = manifest[g_idx]["n_removed"]
                if k == len(g_indices) - 1:
                    inferred_by_gap[g_idx] = new_words[cursor:]
                else:
                    inferred_by_gap[g_idx] = new_words[cursor : cursor + n_exp]
                    cursor += n_exp

    corrections: list[dict] = []
    for entry, inf in zip(manifest, inferred_by_gap):
        n = entry["n_removed"]
        if len(inf) == 0:
            status = "not_filled"
        elif len(inf) == n:
            status = "filled"
        else:
            status = "wrong_length"
        corrections.append(
            {
                "gap_id": entry["gap_id"],
                "inferred": inf,
                "status": status,
            }
        )
    return corrections


def load_manifest_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
