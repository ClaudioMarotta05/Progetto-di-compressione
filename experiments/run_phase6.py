"""Fase 6: compressione lossless dei 3 stream (deflate/bwt/ppm), container gapcomp/1 e verifica ricostruzione."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.codecs import ALGORITHMS, compress_bytes, decompress_bytes
from compression.container import (
    STREAM_NAMES,
    check_manifest_no_oracle,
    pack_container,
    unpack_container,
)
from compression.residuals import (
    decode_residuals,
    encode_residuals,
    reconstruct_from_inference,
    reconstruct_text,
)

BASE = Path(".")
PHASE2_DIR = BASE / "experiments/outputs/phase2"
PHASE5_DIR = BASE / "experiments/outputs/phase5"
PHASE6_DIR = BASE / "experiments/outputs/phase6"
DEV_DIR = BASE / "data/dev"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_baselines(original_bytes: bytes) -> dict[str, dict]:
    out = {}
    for algo in ALGORITHMS:
        comp = compress_bytes(original_bytes, algo)
        dec = decompress_bytes(comp, algo)
        out[algo] = {
            "raw_bytes": len(original_bytes),
            "comp_bytes": len(comp),
            "ratio": len(comp) / len(original_bytes) if original_bytes else 0.0,
            "lossless_verified": dec == original_bytes,
        }
    return out


def process_model(model_dir: Path, doc_id: str, save_containers: bool) -> list[dict]:
    model_id = model_dir.name
    orig_path = DEV_DIR / f"{doc_id}.txt"
    if not orig_path.exists():
        raise FileNotFoundError(f"Documento originale mancante: {orig_path}")

    original_text = orig_path.read_text(encoding="utf-8")
    original_bytes = original_text.encode("utf-8")
    original_sha = sha256_bytes(original_bytes)
    baselines = compute_baselines(original_bytes)

    in_phase2 = PHASE2_DIR / doc_id
    out_model = PHASE6_DIR / model_id
    reports_dir = out_model / "reports"
    containers_dir = out_model / "containers"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if save_containers:
        containers_dir.mkdir(parents=True, exist_ok=True)

    phase5_files = sorted(model_dir.glob("*.json"))
    print(f"\n===== FASE 6 — {model_id} ({len(phase5_files)} combinazioni) =====")
    print(
        f"Originale {doc_id}: {len(original_bytes):,} B | Baseline -> "
        + ", ".join(
            f"{a}={baselines[a]['comp_bytes']:,} B ({baselines[a]['ratio']:.3f})"
            for a in ALGORITHMS
        )
    )

    summary_rows: list[dict] = []
    failures = 0

    for p5_path in phase5_files:
        tag = p5_path.stem
        p5 = json.loads(p5_path.read_text(encoding="utf-8"))
        strategy = p5["strategy"]
        ratio = p5["ratio"]
        condition = p5["condition"]
        model_corrections = p5["corrections"]

        skel_path = in_phase2 / f"{tag}_skeleton.txt"
        man_path = in_phase2 / f"{tag}_manifest.jsonl"
        ora_path = in_phase2 / f"{tag}_oracle.jsonl"

        skeleton_bytes = skel_path.read_bytes()
        manifest_bytes = man_path.read_bytes()
        skeleton_text = skeleton_bytes.decode("utf-8")
        manifest = load_jsonl(man_path)
        oracle = load_jsonl(ora_path)

        # 1. Encoder: calcola i residui lossless rispetto alle predizioni di Fase 5
        residuals_bytes, res_stats = encode_residuals(manifest, model_corrections, oracle)
        oracle_not_in_payload = check_manifest_no_oracle(manifest_bytes)

        # Ricostruzione solo-inferenza (senza residui), utile per distinguere errore inferenza in Fase 7
        inf_only_text = reconstruct_from_inference(strategy, skeleton_text, manifest, model_corrections)
        inf_exact_match_original = inf_only_text == original_text

        algo_results: dict[str, dict] = {}
        for algo in ALGORITHMS:
            # 2. Impacchetta il container compresso gapcomp/1
            blob, _ = pack_container(
                doc_id=doc_id,
                strategy=strategy,
                ratio=ratio,
                condition=condition,
                model_id=model_id,
                algorithm=algo,
                skeleton_bytes=skeleton_bytes,
                manifest_bytes=manifest_bytes,
                corrections_bytes=residuals_bytes,
            )

            if save_containers:
                c_path = containers_dir / f"{tag}.{algo}.gapcomp"
                c_path.write_bytes(blob)
                blob_to_decode = c_path.read_bytes()
            else:
                blob_to_decode = blob

            # 3. Decoder: decomprime dal container e ricostruisce SENZA leggere l'oracle
            header, dec_streams, stream_checks = unpack_container(blob_to_decode)
            streams_lossless = all(stream_checks.values())

            dec_manifest = [
                json.loads(line)
                for line in dec_streams["manifest_or_placeholder"].decode("utf-8").splitlines()
                if line.strip()
            ]
            dec_gap_texts = decode_residuals(
                dec_streams["corrections"], dec_manifest, model_corrections
            )
            reconstructed = reconstruct_text(
                strategy,
                dec_streams["skeleton"].decode("utf-8"),
                dec_manifest,
                dec_gap_texts,
            )
            reconstructed_bytes = reconstructed.encode("utf-8")
            identity_lossless = (
                reconstructed == original_text and sha256_bytes(reconstructed_bytes) == original_sha
            )
            lossless_verified = streams_lossless and identity_lossless and oracle_not_in_payload

            if not lossless_verified:
                failures += 1

            streams_meta = header["streams"]
            comp_per_stream = {s: streams_meta[s]["comp_len"] for s in STREAM_NAMES}
            raw_per_stream = {s: streams_meta[s]["raw_len"] for s in STREAM_NAMES}
            ratio_per_stream = {
                s: (comp_per_stream[s] / raw_per_stream[s] if raw_per_stream[s] else 0.0)
                for s in STREAM_NAMES
            }

            combined_comp_bytes = sum(comp_per_stream.values())
            ratio_combined = combined_comp_bytes / len(original_bytes)
            ratio_container = len(blob_to_decode) / len(original_bytes)
            overhead_manifest_of_combined = (
                raw_per_stream["manifest_or_placeholder"] / raw_per_stream["skeleton"]
                if raw_per_stream["skeleton"]
                else 0.0
            )
            overhead_manifest_of_original = (
                comp_per_stream["manifest_or_placeholder"] / len(original_bytes)
            )
            baseline_comp = baselines[algo]["comp_bytes"]
            vs_baseline = combined_comp_bytes / baseline_comp if baseline_comp else 0.0

            algo_results[algo] = {
                "raw_bytes_per_stream": raw_per_stream,
                "comp_bytes_per_stream": comp_per_stream,
                "ratio_per_stream": {k: round(v, 6) for k, v in ratio_per_stream.items()},
                "combined_comp_bytes": combined_comp_bytes,
                "container_bytes": len(blob_to_decode),
                "baseline_comp_bytes": baseline_comp,
                "ratio_combined": round(ratio_combined, 6),
                "ratio_container": round(ratio_container, 6),
                "vs_baseline": round(vs_baseline, 6),
                "overhead_manifest_or_placeholder": round(overhead_manifest_of_combined, 6),
                "overhead_manifest_vs_original": round(overhead_manifest_of_original, 6),
                "streams_lossless": streams_lossless,
                "identity_lossless": identity_lossless,
                "lossless_verified": lossless_verified,
            }

            summary_rows.append(
                {
                    "model_id": model_id,
                    "doc_id": doc_id,
                    "tag": tag,
                    "strategy": strategy,
                    "ratio_target": ratio,
                    "condition": condition,
                    "algorithm": algo,
                    "n_gaps": res_stats["n_gaps"],
                    "n_exact_hits": res_stats["n_exact_hits"],
                    "exact_hit_rate": res_stats["exact_hit_rate"],
                    "orig_bytes": len(original_bytes),
                    "baseline_comp_bytes": baseline_comp,
                    "skel_comp_bytes": comp_per_stream["skeleton"],
                    "manifest_comp_bytes": comp_per_stream["manifest_or_placeholder"],
                    "corr_comp_bytes": comp_per_stream["corrections"],
                    "combined_comp_bytes": combined_comp_bytes,
                    "container_bytes": len(blob_to_decode),
                    "ratio_combined": round(ratio_combined, 6),
                    "vs_baseline": round(vs_baseline, 6),
                    "overhead_manifest": round(overhead_manifest_of_combined, 6),
                    "lossless_verified": lossless_verified,
                }
            )

        report = {
            "model_id": model_id,
            "hf_id": p5["hf_id"],
            "revision": p5["revision"],
            "doc_id": doc_id,
            "tag": tag,
            "strategy": strategy,
            "ratio_target": ratio,
            "condition": condition,
            "original_bytes": len(original_bytes),
            "original_sha256": original_sha,
            "residual_stats": res_stats,
            "reconstruction_checks": {
                "identity_lossless": all(r["identity_lossless"] for r in algo_results.values()),
                "oracle_not_in_payload": oracle_not_in_payload,
                "compression_error_bytes": 0
                if all(r["lossless_verified"] for r in algo_results.values())
                else -1,
                "inference_only_exact_match_original": inf_exact_match_original,
            },
            "compression": algo_results,
        }
        (reports_dir / f"{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Scrive summary.csv
    if summary_rows:
        csv_path = out_model / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Stampa tabella aggregata per (strategia, algoritmo)
    agg = defaultdict(list)
    for row in summary_rows:
        agg[(row["strategy"], row["algorithm"])].append(row)

    print(
        f"\nVerifica lossless: {'TUTTE OK' if failures == 0 else f'{failures} FALLITE'} "
        f"({len(summary_rows) - failures}/{len(summary_rows)} verifiche)"
    )
    print(
        f"\n{'Strat':5s} {'Algo':7s} | {'Hit%':>6s} | {'RatioComb':>9s} | "
        f"{'vsBase':>7s} | {'ManOv%':>7s} | {'Lossless':>8s}"
    )
    print("-" * 66)
    for (s, a), rows in sorted(agg.items()):
        hit = sum(r["exact_hit_rate"] for r in rows) / len(rows) * 100
        rc = sum(r["ratio_combined"] for r in rows) / len(rows)
        vb = sum(r["vs_baseline"] for r in rows) / len(rows)
        ov = sum(r["overhead_manifest"] for r in rows) / len(rows) * 100
        ok = all(r["lossless_verified"] for r in rows)
        print(f"{s:5s} {a:7s} | {hit:5.2f}% | {rc:9.4f} | {vb:6.3f}x | {ov:6.2f}% | {'OK' if ok else 'FAIL':>8s}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="es. model_1, model_2 oppure all")
    parser.add_argument("--doc-id", default="finewiki_it_small_001")
    parser.add_argument(
        "--no-containers",
        action="store_true",
        help="Non salvare i file binari .gapcomp su disco (verifica comunque in memoria)",
    )
    args = parser.parse_args()

    model_dirs = sorted(p for p in PHASE5_DIR.iterdir() if p.is_dir())
    if args.model != "all":
        model_dirs = [p for p in model_dirs if p.name == args.model]

    if not model_dirs:
        print("Nessuna cartella modello trovata in experiments/outputs/phase5/")
        sys.exit(1)

    for m_dir in model_dirs:
        process_model(m_dir, doc_id=args.doc_id, save_containers=not args.no_containers)


if __name__ == "__main__":
    main()
