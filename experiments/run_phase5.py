"""Fase 5 minimo: inferenza su 1 doc small, 40 combinazioni, 2 modelli."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.prompts import build_prompt
from inference.runner import (
    load_model_and_tokenizer, generate, load_config, set_determinism,
)
from inference.corrections import (
    compute_corrections, load_manifest_jsonl,
)


BASE = Path(".")
INPUT_DIR = BASE / "experiments/outputs/phase2/finewiki_it_small_001"
OUTPUT_DIR = BASE / "experiments/outputs/phase5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_config(BASE / "config/inference.yaml")
set_determinism(cfg["decoding"]["seed"])


def discover_combinations():
    """Trova tutte le coppie (strategy, ratio, condition) dai file skeleton."""
    combos = []
    for skel in sorted(INPUT_DIR.glob("*_skeleton.txt")):
        stem = skel.stem  # es. "A_30_content_band_skeleton"
        parts = stem.replace("_skeleton", "").split("_", 2)
        strategy = parts[0]
        ratio = int(parts[1]) / 100
        condition = parts[2]
        manifest = INPUT_DIR / f"{strategy}_{int(ratio*100):02d}_{condition}_manifest.jsonl"
        if not manifest.exists():
            continue
        combos.append({
            "strategy": strategy,
            "ratio": ratio,
            "condition": condition,
            "skeleton_path": skel,
            "manifest_path": manifest,
        })
    return combos


def main():
    combos = discover_combinations()
    print(f"Trovate {len(combos)} combinazioni.\n")

    for model_cfg in cfg["models"]:
        model_id = model_cfg["id"]
        print(f"\n===== MODELLO {model_id}: {model_cfg['hf_id']} =====")
        model, tok, dev = load_model_and_tokenizer(
            model_cfg["hf_id"], model_cfg["revision"],
        )
        print(f"Device: {dev}\n")

        model_out = OUTPUT_DIR / model_id
        model_out.mkdir(parents=True, exist_ok=True)

        for i, combo in enumerate(combos, start=1):
            tag = f"{combo['strategy']}_{int(combo['ratio']*100):02d}_{combo['condition']}"
            out_path = model_out / f"{tag}.json"
            if out_path.exists():
                print(f"[{i:02d}/{len(combos)}] SKIP {tag} (già fatto)")
                continue

            skeleton = combo["skeleton_path"].read_text(encoding="utf-8")
            manifest = load_manifest_jsonl(combo["manifest_path"])
            prompt = build_prompt(combo["strategy"], skeleton)

            print(f"[{i:02d}/{len(combos)}] {tag}...", end=" ", flush=True)
            output = generate(model, tok, prompt, max_new_tokens=4096)
            corrections = compute_corrections(
                combo["strategy"], skeleton, output, manifest,
            )
            record = {
                "model_id": model_id,
                "hf_id": model_cfg["hf_id"],
                "revision": model_cfg["revision"],
                "strategy": combo["strategy"],
                "ratio": combo["ratio"],
                "condition": combo["condition"],
                "output_text": output,
                "corrections": corrections,
                "n_gaps": len(manifest),
                "n_filled": sum(1 for c in corrections if c["status"] == "filled"),
                "n_wrong_length": sum(1 for c in corrections if c["status"] == "wrong_length"),
                "n_not_filled": sum(1 for c in corrections if c["status"] == "not_filled"),
            }
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"OK ({record['n_filled']}/{record['n_gaps']} filled)")


if __name__ == "__main__":
    main()
