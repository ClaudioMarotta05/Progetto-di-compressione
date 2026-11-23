"""Validazioni Fase 2: determinismo e ricostruzione esatta su tutte le condizioni."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elision.generator import generate, load_config
from elision.tokenizer import tokenize, tokenize_with_placeholders

BASE = Path(".")
config = load_config(BASE / "config/base.yaml")
elision_cfg = load_config(BASE / "config/elision.yaml")
conditions = elision_cfg["gap_generation"]["word_selection"]["conditions"]

doc_path = sorted(Path("data/dev").glob("*small*.txt"))[0]
doc_id = doc_path.stem
text = doc_path.read_text(encoding="utf-8")

errors = 0
warnings = 0
skipped = 0

for strategy in ("A", "B"):
    for ratio in tuple(elision_cfg["ratios"]["values"]):
        for condition in conditions:
            try:
                a = generate(text, doc_id, strategy, ratio, condition, config, BASE)
                b = generate(text, doc_id, strategy, ratio, condition, config, BASE)
            except FileNotFoundError:
                skipped += 1
                print(f"SKIP {condition['name']}/{strategy}/{ratio}")
                continue

            tag = f"{condition['name']}/{strategy}/{ratio}"

            # 1. Determinismo
            if a[3]["skeleton_sha256"] != b[3]["skeleton_sha256"]:
                print(f"FAIL determinismo {tag}")
                errors += 1
                continue

            skeleton, manifest, oracle, meta = a

            # 2. Ricostruzione esatta
            tokens = tokenize_with_placeholders(skeleton) if strategy == "B" else tokenize(skeleton)
            oracle_by_id = {o["gap_id"]: o for o in oracle}

            if strategy == "A":
                insertions = {}
                for g in manifest:
                    insertions.setdefault(g["skeleton_pos"], []).append(
                        oracle_by_id[g["gap_id"]]["removed_text"]
                    )
                parts = []
                i = 0
                while i < len(tokens):
                    if i in insertions:
                        parts.extend(insertions[i])
                    parts.append(tokens[i].text)
                    i += 1
                if len(tokens) in insertions:
                    parts.extend(insertions[len(tokens)])
            else:
                substitutions = {
                    g["skeleton_pos"]: oracle_by_id[g["gap_id"]]["removed_text"]
                    for g in manifest
                }
                parts = []
                i = 0
                while i < len(tokens):
                    if i in substitutions:
                        parts.append(substitutions[i])
                    else:
                        parts.append(tokens[i].text)
                    i += 1

            ricostruito = "".join(parts)
            if ricostruito != text:
                print(f"FAIL ricostruzione {tag}")
                for j, (c1, c2) in enumerate(zip(ricostruito, text)):
                    if c1 != c2:
                        print(f"  diff a {j}: {ricostruito[max(0,j-20):j+20]!r} != {text[max(0,j-20):j+20]!r}")
                        break
                errors += 1

            # 3. Coerenza lunghezze
            if len(manifest) != len(oracle):
                print(f"FAIL lunghezze manifest/oracle {tag}")
                errors += 1

            # 4. Conteggio parole scheletro (A)
            if strategy == "A":
                skel_words = sum(1 for t in tokenize(skeleton) if t.is_word)
                expected = meta["n_words_total"] - meta["n_words_removed"]
                if skel_words != expected:
                    print(f"FAIL conteggio parole {tag}: {skel_words} vs {expected}")
                    errors += 1

            # 5. Effective ratio (warning)
            if meta["target_reachable"]:
                if abs(meta["effective_ratio"] - ratio) > 0.10:
                    print(f"WARN effective_ratio {tag}: {meta['effective_ratio']:.2f} vs {ratio}")
                    warnings += 1
                else:
                    print(f"CAP {tag}: tetto strutturale {meta['candidate_ratio']:.2f} "
                        f"(target {ratio}), elise {meta['effective_ratio']:.2f}")

print(f"\nValidazione completata: {errors} errori, {warnings} warning, {skipped} skip")
sys.exit(1 if errors else 0)
