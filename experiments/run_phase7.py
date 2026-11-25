"""Fase 7: valutazione completa (fedeltà, errori, controlli e verifica ricostruzione)."""
import sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

from evaluation.controls import (
    compute_context_density_per_gap,
    compute_elided_composition,
    correlate_context_with_accuracy,
    load_external_lexicons,
)
from evaluation.errors import evaluate_errors
from evaluation.fidelity import compute_bleu, compute_precision_at_k, compute_rouge_l

BASE_DIR = Path(".")
DOC_ID = "finewiki_it_small_001"
def _find_doc(base_dir: Path, doc_id: str) -> Path:
    hits = sorted((base_dir / "data").rglob(f"{doc_id}.txt")) or sorted(base_dir.rglob(f"{doc_id}.txt"))
    if not hits:
        raise FileNotFoundError(f"Documento {doc_id}.txt non trovato in {base_dir}")
    for h in hits:
        if "dev" in h.parts:
            return h
    return hits[0]

DOC_PATH = _find_doc(BASE_DIR, DOC_ID)
PHASE2_DIR = BASE_DIR / "experiments/outputs/phase2" / DOC_ID
PHASE5_DIR = BASE_DIR / "experiments/outputs/phase5"
PHASE6_DIR = BASE_DIR / "experiments/outputs/phase6"
PHASE7_DIR = BASE_DIR / "experiments/outputs/phase7"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_model(model_id: str, original_text: str, elision_cfg: dict, metrics_cfg: dict, lexicons) -> List[dict]:
    p5_model_dir = PHASE5_DIR / model_id
    p6_reports_dir = PHASE6_DIR / model_id / "reports"
    out_model_dir = PHASE7_DIR / model_id
    out_reports_dir = out_model_dir / "reports"
    out_reports_dir.mkdir(parents=True, exist_ok=True)

    k_values = metrics_cfg["fidelity"]["precision_at_k"]["k_values"]
    window_words = metrics_cfg["controls"]["context_density"]["window_words"]
    anchor_params = next(
        c["params"]
        for c in elision_cfg["gap_generation"]["word_selection"]["conditions"]
        if c["name"] == "anchor_preservation"
    )

    p5_files = sorted(p5_model_dir.glob("*.json"))
    if not p5_files:
        raise FileNotFoundError(f"Nessun output di Fase 5 trovato in {p5_model_dir}")

    rows: List[dict] = []
    gap_level_rows: List[dict] = []
    all_hyps: List[str] = []
    all_refs: List[str] = []

    for p5_path in p5_files:
        tag = p5_path.stem
        p5 = json.loads(p5_path.read_text(encoding="utf-8"))
        manifest = load_jsonl(PHASE2_DIR / f"{tag}_manifest.jsonl")
        oracle = load_jsonl(PHASE2_DIR / f"{tag}_oracle.jsonl")
        p6_path = p6_reports_dir / f"{tag}.json"
        p6 = json.loads(p6_path.read_text(encoding="utf-8")) if p6_path.exists() else None

        output_text = p5.get("output_text", "")
        predictions = p5.get("corrections", [])

        all_hyps.append(output_text)
        all_refs.append(original_text)

        bleu = compute_bleu([output_text], [original_text])
        rouge = compute_rouge_l(output_text, original_text)
        p_at_k = compute_precision_at_k(predictions, oracle, k_values=k_values)
        err_summary, per_gap_errs = evaluate_errors(predictions, oracle)

        per_gap_density, density_stats = compute_context_density_per_gap(
            original_text=original_text,
            manifest=manifest,
            lexicons=lexicons,
            window_words=window_words,
            anchor_top_k=anchor_params.get("anchor_top_k_per_window", 3),
            anchor_window=anchor_params.get("window_words", 50),
        )
        elided_comp = compute_elided_composition(oracle, lexicons)
        density_corr = correlate_context_with_accuracy(per_gap_density, per_gap_errs)

        lossless_ok = False
        oracle_not_in_payload = False
        ppm_ratio = None
        bwt_ratio = None
        deflate_ratio = None
        ppm_vs_base = None
        if p6:
            comp = p6.get("compression", {})
            lossless_ok = bool(comp) and all(bool(v.get("lossless_verified")) for v in comp.values())
            oracle_not_in_payload = bool(p6.get("reconstruction_checks", {}).get("oracle_not_in_payload", False))
            if "ppm" in comp:
                ppm_ratio = comp["ppm"].get("ratio_combined")
                ppm_vs_base = comp["ppm"].get("vs_baseline")
            if "bwt" in comp:
                bwt_ratio = comp["bwt"].get("ratio_combined")
            if "deflate" in comp:
                deflate_ratio = comp["deflate"].get("ratio_combined")

        reconstruction_checks = {
            "identity_lossless": lossless_ok,
            "quality_vs_oracle_computed": True,
            "compression_error_bytes": 0 if lossless_ok else None,
            "inference_error_rate": 1.0 - err_summary["rates"]["accuracy"],
            "oracle_not_in_payload": oracle_not_in_payload,
        }

        report = {
            "model_id": model_id,
            "hf_id": p5.get("hf_id"),
            "doc_id": DOC_ID,
            "tag": tag,
            "strategy": p5["strategy"],
            "ratio_target": float(p5["ratio"]),
            "condition": p5["condition"],
            "fidelity": {
                "bleu": bleu,
                "rouge_l": rouge,
                "precision_at_k": p_at_k,
            },
            "errors": err_summary,
            "controls": {
                "context_density": {**density_stats, "correlation_with_accuracy": density_corr},
                "elided_composition": elided_comp,
            },
            "reconstruction": reconstruction_checks,
        }
        (out_reports_dir / f"{tag}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        row = {
            "model_id": model_id,
            "tag": tag,
            "strategy": p5["strategy"],
            "ratio_target": float(p5["ratio"]),
            "condition": p5["condition"],
            "n_gaps": err_summary["n_gaps"],
            "bleu": bleu["bleu"],
            "rouge_l_f1": rouge["rouge_l_f1"],
            "precision_at_1": p_at_k["precision_at_1"],
            "precision_at_3": p_at_k["precision_at_3"],
            "precision_at_5": p_at_k["precision_at_5"],
            "precision_at_10": p_at_k["precision_at_10"],
            "token_accuracy": err_summary["rates"]["accuracy"],
            "exact_gap_accuracy": err_summary["rates"]["exact_gap_accuracy"],
            "hallucination_rate": err_summary["rates"]["hallucination_rate"],
            "omission_rate": err_summary["rates"]["omission_rate"],
            "length_error": err_summary["rates"]["length_error"],
            "overinsertion_rate": err_summary["rates"]["overinsertion_rate"],
            "wrong_length_gap_rate": err_summary["rates"]["wrong_length_gap_rate"],
            "mean_context_density": density_stats["mean_context_density"],
            "density_acc_spearman_r": density_corr["spearman_r"],
            "density_acc_spearman_p": density_corr["spearman_p"],
            "elided_content_ratio": elided_comp["content_ratio"],
            "elided_function_word_ratio": elided_comp["function_word_ratio"],
            "elided_mean_zipf_rank": elided_comp["mean_zipf_rank"],
            "ratio_combined_ppm": ppm_ratio,
            "ratio_combined_bwt": bwt_ratio,
            "ratio_combined_deflate": deflate_ratio,
            "vs_baseline_ppm": ppm_vs_base,
            "identity_lossless": lossless_ok,
            "oracle_not_in_payload": oracle_not_in_payload,
        }
        rows.append(row)

        for g in per_gap_errs:
            gid = int(g["gap_id"])
            gap_level_rows.append(
                {
                    "model_id": model_id,
                    "strategy": p5["strategy"],
                    "ratio_target": float(p5["ratio"]),
                    "condition": p5["condition"],
                    "gap_id": gid,
                    "context_density": per_gap_density.get(gid, 0.0),
                    "gap_accuracy": g["gap_accuracy"],
                    "exact_match": g["exact_match"],
                }
            )

    summary_csv = out_model_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gaps_csv = out_model_dir / "gap_context_accuracy.csv"
    with gaps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_level_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gap_level_rows)

    corpus_bleu = compute_bleu(all_hyps, all_refs)
    print(f"\n===== FASE 7 — {model_id} ({len(rows)} combinazioni) =====")
    print(f"sacrebleu corpus-level aggregato: {corpus_bleu['bleu']:.2f}")
    print(
        f"{'Strat':<5} | {'BLEU':>6} | {'ROUGE-L':>7} | {'TokAcc':>7} | {'Exact%':>7} | "
        f"{'Halluc':>7} | {'Omiss':>7} | {'P@10':>6} | {'CtxDns':>6}"
    )
    print("-" * 82)
    for strat in ("A", "B"):
        sub = [r for r in rows if r["strategy"] == strat]
        if not sub:
            continue
        avg = lambda k: sum(r[k] for r in sub if r[k] is not None) / len([r for r in sub if r[k] is not None])
        print(
            f"{strat:<5} | {avg('bleu'):6.2f} | {avg('rouge_l_f1'):7.4f} | "
            f"{avg('token_accuracy')*100:6.2f}% | {avg('exact_gap_accuracy')*100:6.2f}% | "
            f"{avg('hallucination_rate')*100:6.2f}% | {avg('omission_rate')*100:6.2f}% | "
            f"{avg('precision_at_10')*100:5.1f}% | {avg('mean_context_density'):6.4f}"
        )

    print("\nControllo elided_composition per condizione (media su A+B, tutti i ratio):")
    print(f"{'Condizione':<22} | {'Content%':>9} | {'Function%':>9} | {'MeanZipfRank':>12} | {'CtxDensity':>10}")
    print("-" * 74)
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        avg = lambda k: sum(r[k] for r in sub) / len(sub)
        print(
            f"{cond:<22} | {avg('elided_content_ratio')*100:8.2f}% | "
            f"{avg('elided_function_word_ratio')*100:8.2f}% | "
            f"{avg('elided_mean_zipf_rank'):12.1f} | {avg('mean_context_density'):10.4f}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fase 7: valutazione qualità, errori e controlli")
    parser.add_argument("--model", default="all", help="model_1, model_2 oppure all")
    args = parser.parse_args()

    elision_cfg = yaml.safe_load((BASE_DIR / "config/elision.yaml").read_text(encoding="utf-8"))
    metrics_cfg = yaml.safe_load((BASE_DIR / "config/metrics.yaml").read_text(encoding="utf-8"))
    original_text = DOC_PATH.read_text(encoding="utf-8")
    lexicons = load_external_lexicons(BASE_DIR, elision_cfg)

    if args.model == "all":
        models = sorted(p.name for p in PHASE5_DIR.iterdir() if p.is_dir() and list(p.glob("*.json")))
    else:
        models = [args.model]

    all_rows: List[dict] = []
    for m in models:
        all_rows.extend(evaluate_model(m, original_text, elision_cfg, metrics_cfg, lexicons))

    if all_rows:
        PHASE7_DIR.mkdir(parents=True, exist_ok=True)
        combined = PHASE7_DIR / "combined_summary.csv"
        with combined.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nTabella unificata salvata in: {combined}")


if __name__ == "__main__":
    main()
