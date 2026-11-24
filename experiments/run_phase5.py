"""Fase 5 minimo: inferenza su finewiki_it_small_001 (40 combinazioni x 2 modelli)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.prompts import build_prompt
from inference.runner import (
    load_model_and_tokenizer,
    generate_chunked,
    load_config,
    set_determinism,
)
from inference.corrections import compute_corrections, load_manifest_jsonl

BASE = Path(".")
INPUT_DIR = BASE / "experiments/outputs/phase2/finewiki_it_small_001"
OUTPUT_DIR = BASE / "experiments/outputs/phase5"


def discover_combinations(filter_tags: set[str] | None = None) -> list[dict]:
    combos = []
    for skel in sorted(INPUT_DIR.glob("*_skeleton.txt")):
        tag = skel.stem.replace("_skeleton", "")
        if filter_tags and tag not in filter_tags:
            continue
        parts = tag.split("_", 2)
        strategy = parts[0]
        ratio = int(parts[1]) / 100
        condition = parts[2]
        manifest = INPUT_DIR / f"{tag}_manifest.jsonl"
        if not manifest.exists():
            continue
        combos.append(
            {
                "tag": tag,
                "strategy": strategy,
                "ratio": ratio,
                "condition": condition,
                "skeleton_path": skel,
                "manifest_path": manifest,
            }
        )
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        help="id del modello in config/inference.yaml (es. model_1, model_2) oppure 'all'",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Esegui solo questi tag (es. A_10_anchor_preservation B_10_anchor_preservation)",
    )
    parser.add_argument("--chunk-words", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(BASE / "config/inference.yaml")
    set_determinism(cfg["decoding"]["seed"])

    filter_tags = set(args.tags) if args.tags else None
    combos = discover_combinations(filter_tags)
    print(f"Trovate {len(combos)} combinazioni.\n")

    models = [
        m for m in cfg["models"] if args.model == "all" or m["id"] == args.model
    ]

    for model_cfg in models:
        model_id = model_cfg["id"]
        print(f"\n===== MODELLO {model_id}: {model_cfg['hf_id']} =====")
        model, tok, dev = load_model_and_tokenizer(
            model_cfg["hf_id"], model_cfg["revision"]
        )
        print(f"Device: {dev}\n")

        model_out = OUTPUT_DIR / model_id
        model_out.mkdir(parents=True, exist_ok=True)

        for i, combo in enumerate(combos, start=1):
            tag = combo["tag"]
            out_path = model_out / f"{tag}.json"
            if out_path.exists():
                print(f"[{i:02d}/{len(combos)}] SKIP {tag} (già fatto)")
                continue

            skeleton = combo["skeleton_path"].read_text(encoding="utf-8")
            manifest = load_manifest_jsonl(combo["manifest_path"])

            print(f"[{i:02d}/{len(combos)}] {tag}...", flush=True)
            t0 = time.time()
            output = generate_chunked(
                model,
                tok,
                combo["strategy"],
                skeleton,
                build_prompt,
                ratio=combo["ratio"],
                max_words_per_chunk=args.chunk_words,
                verbose=True,
            )
            dt = time.time() - t0
            corrections = compute_corrections(
                combo["strategy"], skeleton, output, manifest
            )

            record = {
                "model_id": model_id,
                "hf_id": model_cfg["hf_id"],
                "revision": model_cfg["revision"],
                "strategy": combo["strategy"],
                "ratio": combo["ratio"],
                "condition": combo["condition"],
                "elapsed_sec": round(dt, 1),
                "output_text": output,
                "corrections": corrections,
                "n_gaps": len(manifest),
                "n_filled": sum(1 for c in corrections if c["status"] == "filled"),
                "n_wrong_length": sum(
                    1 for c in corrections if c["status"] == "wrong_length"
                ),
                "n_not_filled": sum(
                    1 for c in corrections if c["status"] == "not_filled"
                ),
            }
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"  -> OK in {dt/60:.1f} min "
                f"(filled={record['n_filled']}, wrong_len={record['n_wrong_length']}, "
                f"not_filled={record['n_not_filled']} / {record['n_gaps']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
