"""Esegue la Fase 2 su un documento di prova e salva gli output."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elision.generator import generate, load_config

BASE = Path(".")
config = load_config(BASE / "config/base.yaml")
elision_cfg = load_config(BASE / "config/elision.yaml")

doc_path = sorted(Path("data/dev").glob("*small*.txt"))[0]
doc_id = doc_path.stem
text = doc_path.read_text(encoding="utf-8")

out_dir = Path("experiments/outputs/phase2") / doc_id
out_dir.mkdir(parents=True, exist_ok=True)

for strategy in ("A", "B"):
    for ratio in tuple(elision_cfg["ratios"]["values"]):
        for condition in elision_cfg["gap_generation"]["word_selection"]["conditions"]:
            try:
                skeleton, manifest, oracle, meta = generate(
                    text, doc_id, strategy, ratio, condition, config, BASE
                )
            except FileNotFoundError as e:
                print(f"SKIP {condition['name']} (file mancante: {e})")
                continue
            tag = f"{strategy}_{int(ratio*100):02d}_{condition['name']}"
            (out_dir / f"{tag}_skeleton.txt").write_text(skeleton, encoding="utf-8")
            (out_dir / f"{tag}_manifest.jsonl").write_text(
                "\n".join(json.dumps(m) for m in manifest), encoding="utf-8"
            )
            (out_dir / f"{tag}_oracle.jsonl").write_text(
                "\n".join(json.dumps(o) for o in oracle), encoding="utf-8"
            )
            (out_dir / f"{tag}_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"Generato {tag}")
