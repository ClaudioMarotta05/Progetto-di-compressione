"""Calcolo dei residui lossless rispetto alle predizioni del modello e ricostruzione testo."""
from elision.tokenizer import tokenize, tokenize_with_placeholders


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append("\\" + nxt)
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def predict_gap_text(gap: dict, inferred_words: list[str]) -> str:
    """Converte deterministicamente le parole predette dal modello nel testo del gap.
    Usa solo informazioni presenti nel manifest (noto sia a encoder sia a decoder).
    In elision/generator.py i gap con orig_token_start > 0 assorbono lo spazio precedente.
    """
    if not inferred_words:
        return ""
    prefix = " " if gap["orig_token_start"] > 0 else ""
    return prefix + " ".join(inferred_words)


def encode_residuals(
    manifest: list[dict],
    model_corrections: list[dict],
    oracle: list[dict],
) -> tuple[bytes, dict]:
    """Lato encoder: confronta le predizioni del modello con l'oracle e produce lo stream residuo.
    Formato (UTF-8, una riga per gap in ordine di manifest):
      - '='       -> exact hit: il decoder usa direttamente la predizione del modello
      - '+<text>' -> miss/errore: testo correttivo escapato
    """
    pred_by_id = {c["gap_id"]: c["inferred"] for c in model_corrections}
    oracle_by_id = {o["gap_id"]: o["removed_text"] for o in oracle}

    lines: list[str] = []
    n_exact_hits = 0
    n_word_exact_hits = 0

    for g in manifest:
        gid = g["gap_id"]
        inferred_words = pred_by_id.get(gid, [])
        pred_text = predict_gap_text(g, inferred_words)
        true_text = oracle_by_id[gid]

        true_words = [t.text for t in tokenize(true_text) if t.is_word]
        if inferred_words == true_words:
            n_word_exact_hits += 1

        if pred_text == true_text:
            lines.append("=")
            n_exact_hits += 1
        else:
            lines.append("+" + _escape(true_text))

    payload = "\n".join(lines).encode("utf-8")
    n_gaps = len(manifest)
    stats = {
        "n_gaps": n_gaps,
        "n_exact_hits": n_exact_hits,
        "n_word_exact_hits": n_word_exact_hits,
        "exact_hit_rate": round(n_exact_hits / n_gaps, 6) if n_gaps else 0.0,
        "word_exact_hit_rate": round(n_word_exact_hits / n_gaps, 6) if n_gaps else 0.0,
    }
    return payload, stats


def decode_residuals(
    residuals_bytes: bytes,
    manifest: list[dict],
    model_corrections: list[dict],
) -> dict[int, str]:
    """Lato decoder: ricostruisce il testo esatto di ogni gap SENZA accedere all'oracle."""
    if not manifest:
        return {}

    lines = residuals_bytes.decode("utf-8").split("\n")
    if len(lines) != len(manifest):
        raise ValueError(
            f"Numero di righe nel residuo ({len(lines)}) != numero di gap ({len(manifest)})"
        )

    pred_by_id = {c["gap_id"]: c["inferred"] for c in model_corrections}
    gap_texts: dict[int, str] = {}

    for g, line in zip(manifest, lines):
        gid = g["gap_id"]
        if line == "=":
            gap_texts[gid] = predict_gap_text(g, pred_by_id.get(gid, []))
        elif line.startswith("+"):
            gap_texts[gid] = _unescape(line[1:])
        else:
            raise ValueError(f"Riga residuo malformata per gap {gid}: {line!r}")

    return gap_texts


def reconstruct_text(
    strategy: str,
    skeleton: str,
    manifest: list[dict],
    gap_texts: dict[int, str],
) -> str:
    """Ricostruisce il testo completo da scheletro + manifest + testo dei singoli gap."""
    if strategy == "B":
        skel_tokens = tokenize_with_placeholders(skeleton)
        pos_to_text = {g["skeleton_pos"]: gap_texts[g["gap_id"]] for g in manifest}
        out: list[str] = []
        for idx, tok in enumerate(skel_tokens):
            if idx in pos_to_text:
                out.append(pos_to_text[idx])
            else:
                out.append(tok.text)
        return "".join(out)

    if strategy == "A":
        skel_tokens = tokenize(skeleton)
        pos_to_texts: dict[int, list[str]] = {}
        for g in manifest:
            pos_to_texts.setdefault(g["skeleton_pos"], []).append(gap_texts[g["gap_id"]])
        out = []
        for idx in range(len(skel_tokens) + 1):
            if idx in pos_to_texts:
                out.extend(pos_to_texts[idx])
            if idx < len(skel_tokens):
                out.append(skel_tokens[idx].text)
        return "".join(out)

    raise ValueError(f"Strategia sconosciuta: {strategy!r}")


def reconstruct_from_inference(
    strategy: str,
    skeleton: str,
    manifest: list[dict],
    model_corrections: list[dict],
) -> str:
    """Ricostruzione usando SOLO le predizioni del modello (senza residui), utile per Fase 7."""
    pred_by_id = {c["gap_id"]: c["inferred"] for c in model_corrections}
    gap_texts = {
        g["gap_id"]: predict_gap_text(g, pred_by_id.get(g["gap_id"], []))
        for g in manifest
    }
    return reconstruct_text(strategy, skeleton, manifest, gap_texts)
