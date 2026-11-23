"""
experiments/prepare_corpora.py

Genera N documenti per scala (small, medium, large) leggendo in streaming il
subset italiano di finewiki (HuggingFaceFW/finewiki, config itwiki, licenza
CC BY-SA). Ogni documento ha una dimensione target in parole secondo il
tokenizzatore congelato in config/tokenizer.yaml.

Deterministico: due esecuzioni con lo stesso GLOBAL_SEED producono gli stessi
articoli, nello stesso ordine, negli stessi documenti.

Output: N file .txt per scala in data/raw/.
"""

import re
import hashlib
from pathlib import Path
from datasets import load_dataset

# --- Configurazione congelata ---
GLOBAL_SEED = 42
SCALES = {
    "small": 10_000,
    "medium": 100_000,
    "large": 1_000_000,
}
DOCS_PER_SCALE = 10
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
OUTPUT_DIR = Path("data/raw")

def count_words(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carico finewiki/it in streaming...")
    ds = load_dataset(
        "HuggingFaceFW/finewiki",
        name="it",
        split="train",
        streaming=True,
    )
    # Shuffle deterministico: lo stesso GLOBAL_SEED produce lo stesso ordine.
    ds = ds.shuffle(seed=GLOBAL_SEED, buffer_size=10_000)
    iterator = iter(ds)

    total_docs = 0
    for scale_name, target_words in SCALES.items():
        print(f"\n=== Scala {scale_name} (target {target_words} parole/doc) ===")
        for doc_idx in range(1, DOCS_PER_SCALE + 1):
            buffer_texts = []
            buffer_words = 0
            while buffer_words < target_words:
                try:
                    example = next(iterator)
                except StopIteration:
                    print(f"  Dataset esaurito a doc {doc_idx} di {scale_name}")
                    return
                text = example.get("text") or ""
                if not text:
                    continue
                buffer_texts.append(text)
                buffer_words += count_words(text)

            doc_id = f"finewiki_it_{scale_name}_{doc_idx:03d}"
            out_path = OUTPUT_DIR / f"{doc_id}.txt"
            out_path.write_text("\n\n".join(buffer_texts), encoding="utf-8")
            total_docs += 1
            print(f"  {doc_id}: {buffer_words} parole -> {out_path}")

    print(f"\nTotale documenti generati: {total_docs}")
    print(f"Posizione: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
