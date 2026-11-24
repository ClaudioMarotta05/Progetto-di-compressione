"""Allineamento output modello -> gap dello scheletro.

Per strategia B: i placeholder [N] sono ancore affidabili.
Per strategia A: allineamento tramite sottosequenza (SequenceMatcher).
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path


PLACEHOLDER = re.compile(r"\[(\d+)\]")


def align_strategy_b(skeleton: str, output: str, manifest: list[dict]) -> list[dict]:
    """Per B: sostituisce ogni [N] nell'output con le N parole che il modello
    ha messo al suo posto. Se il modello ha lasciato il placeholder, il gap
    risulta non riempito."""
    corrections = []
    for entry in manifest:
        gap_id = entry["gap_id"]
        n = entry["n_removed"]
        # Trova l'n-esimo placeholder nell'output (approssimazione: il primo [N]
        # con N corretto non ancora consumato)
        pattern = re.compile(rf"\[{n}\]")
        match = pattern.search(output)
        if not match:
            corrections.append({
                "gap_id": gap_id,
                "inferred": [],
                "status": "not_filled",
            })
        else:
            # Prendi le N parole successive al placeholder (euristica)
            after = output[match.end():].strip()
            tokens_after = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", after)[:n]
            corrections.append({
                "gap_id": gap_id,
                "inferred": tokens_after,
                "status": "filled" if len(tokens_after) == n else "wrong_length",
            })
    return corrections


def align_strategy_a(skeleton: str, output: str, manifest: list[dict]) -> list[dict]:
    """Per A: allinea skeleton e output con SequenceMatcher.
    Le parole presenti in output ma non in skeleton vicine alla posizione
    del gap sono considerate "inferite" per quel gap."""
    sm = SequenceMatcher(None, skeleton, output, autojunk=False)
    corrections = []
    for entry in manifest:
        gap_pos_char = entry["skeleton_pos"]
        n = entry["n_removed"]
        # Trova il blocco di matching che contiene la posizione del gap
        inferred = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("insert", "replace") and i1 <= gap_pos_char <= i2:
                chunk = output[j1:j2]
                words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", chunk)
                inferred = words
                break
        corrections.append({
            "gap_id": entry["gap_id"],
            "inferred": inferred,
            "status": "filled" if len(inferred) == n else "wrong_length",
        })
    return corrections


def compute_corrections(
    strategy: str,
    skeleton: str,
    output: str,
    manifest: list[dict],
) -> list[dict]:
    if strategy == "A":
        return align_strategy_a(skeleton, output, manifest)
    if strategy == "B":
        return align_strategy_b(skeleton, output, manifest)
    raise ValueError(f"Strategia sconosciuta: {strategy!r}")


def load_manifest_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
