"""Derivazione deterministica del seed per documento."""

import hashlib


def seed_doc(global_seed: int, doc_id: str) -> int:
    """Restituisce un seed intero derivato da global_seed e doc_id.

    Usa SHA-256, mai hash() di Python (randomizzato per processo).
    """
    h = hashlib.sha256(f"{global_seed}:{doc_id}".encode("utf-8")).hexdigest()
    return int(h, 16)
