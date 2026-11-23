"""experiments/build_external_lists.py

Genera le liste esterne (stopword, frequenza, IDF) dal corpus dev.
Le liste sono congelate e hashate per la riproducibilità.

- it_stopwords.txt: lista statica di stopword italiane
- it_frequency_list.txt: parole ordinate per frequenza decrescente (rango 1 = più frequente)
- it_idf.txt: IDF calcolato sui documenti dev
"""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
EXTERNAL = Path("data/external")
LOGS = Path("logs")
EXTERNAL.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Stopword italiane di base (lista statica, congelata)
STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "ed", "o", "od", "ma", "che", "chi", "cui", "non",
    "più", "meno", "molto", "poco", "già", "ancora", "sempre",
    "mai", "anche", "come", "dove", "quando", "perché", "se",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "questo", "questa", "questi", "queste", "quello", "quella",
    "quelli", "quelle", "sono", "sei", "è", "siamo", "siete",
    "sia", "essere", "avere", "ha", "ho", "hai", "abbiamo",
    "avete", "hanno", "loro", "suo", "sua", "suoi", "sue",
    "mio", "mia", "miei", "mie", "tuo", "tua", "tuoi", "tue",
}

# 1. Stopword
(EXTERNAL / "it_stopwords.txt").write_text(
    "\n".join(sorted(STOPWORDS)), encoding="utf-8")
print(f"it_stopwords.txt: {len(STOPWORDS)} parole")

# 2. Frequenza (contata su tutti i documenti dev)
counter: Counter[str] = Counter()
docs_tokens: list[set[str]] = []
for p in sorted(Path("data/dev").glob("*.txt")):
    text = p.read_text(encoding="utf-8").lower()
    words = [m.group() for m in WORD.finditer(text)]
    counter.update(words)
    docs_tokens.append(set(words))

freq_sorted = [w for w, _ in counter.most_common()]
(EXTERNAL / "it_frequency_list.txt").write_text(
    "\n".join(freq_sorted), encoding="utf-8")
print(f"it_frequency_list.txt: {len(freq_sorted)} parole")

# 3. IDF calcolato sul corpus dev
N = len(docs_tokens)
idf_lines = []
for w, _ in counter.most_common():
    df = sum(1 for d in docs_tokens if w in d)
    idf = math.log((N + 1) / (df + 1)) + 1.0
    idf_lines.append(f"{w} {idf:.6f}")
(EXTERNAL / "it_idf.txt").write_text("\n".join(idf_lines), encoding="utf-8")
print(f"it_idf.txt: {len(idf_lines)} parole")

# 4. Hash SHA-256 delle liste esterne
lines = []
for name in ("it_stopwords.txt", "it_frequency_list.txt", "it_idf.txt"):
    p = EXTERNAL / name
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(f"{h}  {p}")
(LOGS / "external_hashes.sha256").write_text("\n".join(lines), encoding="utf-8")
print("logs/external_hashes.sha256 scritto")
