"""
experiments/split_and_manifest.py

Applica la regola di split document_level_hash (config/base.yaml:splits),
sposta i documenti da data/raw/ a data/dev/ o data/holdout/, genera i
manifest.json per ciascuno split e calcola logs/data_hashes.sha256.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
HOLDOUT_FRACTION = 0.20
RAW = Path("data/raw")
DEV = Path("data/dev")
HOLDOUT = Path("data/holdout")
LOGS = Path("logs")

def is_holdout(doc_id: str) -> bool:
    h = hashlib.sha256(f"split:{doc_id}".encode()).hexdigest()
    return int(h, 16) % 100 < int(HOLDOUT_FRACTION * 100)

def n_words(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def main():
    DEV.mkdir(parents=True, exist_ok=True)
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    dev_entries, holdout_entries = [], []
    for src in sorted(RAW.glob("*.txt")):
        doc_id = src.stem
        text = src.read_text(encoding="utf-8")
        nw = n_words(text)
        # deduci la scala dal nome
        parts = doc_id.split("_")
        scale = parts[2]  # "small" | "medium" | "large"
        entry = {
            "doc_id": doc_id,
            "scale": scale,
            "n_words": nw,
            "sha256": sha256_file(src),
        }
        if is_holdout(doc_id):
            dst = HOLDOUT / src.name
            shutil.move(str(src), str(dst))
            holdout_entries.append(entry)
        else:
            dst = DEV / src.name
            shutil.move(str(src), str(dst))
            dev_entries.append(entry)

    (DEV / "manifest.json").write_text(
        json.dumps(dev_entries, indent=2), encoding="utf-8")
    (HOLDOUT / "manifest.json").write_text(
        json.dumps(holdout_entries, indent=2), encoding="utf-8")

    print(f"Dev: {len(dev_entries)} documenti")
    print(f"Holdout: {len(holdout_entries)} documenti")

    # Statistica per scala
    for scale in ("small", "medium", "large"):
        d = sum(1 for e in dev_entries if e["scale"] == scale)
        h = sum(1 for e in holdout_entries if e["scale"] == scale)
        print(f"  {scale}: dev={d}, holdout={h}")

    # Hash SHA-256 di tutti i file in dev e holdout
    out = LOGS / "data_hashes.sha256"
    with out.open("w", encoding="utf-8") as f:
        for split_dir in (DEV, HOLDOUT):
            for p in sorted(split_dir.glob("*")):
                if p.is_file():
                    f.write(f"{sha256_file(p)}  {p}\n")
    print(f"Hash scritti in {out}")

if __name__ == "__main__":
    main()
