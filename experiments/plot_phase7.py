"""Genera i 4 grafici principali della Fase 7 in italiano (300 DPI) per la relazione."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PHASE6_DIR = Path("experiments/outputs/phase6")
PHASE7_DIR = Path("experiments/outputs/phase7")
FIG_DIR = PHASE7_DIR / "figures"

MODEL_LABELS = {
    "model_1": "Qwen2.5-0.5B",
    "model_2": "Qwen2.5-1.5B",
}
COND_LABELS = {
    "anchor_preservation": "Anchor\npreservation",
    "content_band": "Content\nband",
    "function_only": "Function\nonly",
    "random_stratified": "Random\nstratified",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_compression_vs_baseline(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    algos = ["ppm", "bwt", "deflate"]
    algo_names = {"ppm": "PPM", "bwt": "BWT (bz2)", "deflate": "Deflate (zlib)"}
    baselines = {"ppm": 23722 / 117304, "bwt": 26426 / 117304, "deflate": 30201 / 117304}

    x = range(len(algos))
    width = 0.26

    for ax, strat in zip(axes, ("A", "B")):
        sub = df[df["strategy"] == strat]
        m1 = [sub[sub["model_id"] == "model_1"][f"ratio_combined_{a}"].mean() for a in algos]
        m2 = [sub[sub["model_id"] == "model_2"][f"ratio_combined_{a}"].mean() for a in algos]
        base = [baselines[a] for a in algos]

        ax.bar([i - width for i in x], base, width, label="Baseline diretta", color="#4C78A8")
        ax.bar(list(x), m1, width, label=MODEL_LABELS["model_1"], color="#F58518")
        ax.bar([i + width for i in x], m2, width, label=MODEL_LABELS["model_2"], color="#54A24B")

        for i in x:
            ax.text(i - width, base[i] + 0.008, f"{base[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i, m1[i] + 0.008, f"{m1[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width, m2[i] + 0.008, f"{m2[i]:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"Strategia {strat}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([algo_names[a] for a in algos])
        ax.set_ylim(0, 0.60)
        if strat == "A":
            ax.set_ylabel("Rapporto di compressione combinato (comp / orig)")
        ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle("Compressione lossless combinata vs Baseline diretta (media su 40 combinazioni)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_compressione_vs_baseline.png")
    plt.close(fig)


def plot_fidelity_by_ratio(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    metrics = [
        ("bleu", "sacreBLEU (corpus-level)"),
        ("rouge_l_f1", "ROUGE-L (F1)"),
        ("token_accuracy", "Accuratezza token sui gap (%)"),
    ]
    styles = {
        ("model_1", "A"): ("#F58518", "-", "o", "0.5B — Strat. A"),
        ("model_1", "B"): ("#F58518", "--", "s", "0.5B — Strat. B"),
        ("model_2", "A"): ("#54A24B", "-", "o", "1.5B — Strat. A"),
        ("model_2", "B"): ("#54A24B", "--", "s", "1.5B — Strat. B"),
    }

    for ax, (col, ylabel) in zip(axes, metrics):
        for (m, strat), (color, ls, marker, label) in styles.items():
            sub = df[(df["model_id"] == m) & (df["strategy"] == strat)]
            if sub.empty:
                continue
            grp = sub.groupby("ratio_target")[col].mean().sort_index()
            vals = grp.values * 100 if col == "token_accuracy" else grp.values
            ax.plot(grp.index * 100, vals, color=color, linestyle=ls, marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Rapporto di elisione target (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.legend(fontsize=8)

    fig.suptitle("Andamento della fedeltà di ricostruzione al crescere del rapporto di elisione", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_fedelta_per_ratio.png")
    plt.close(fig)


def plot_errors_by_condition(df: pd.DataFrame):
    conds = ["function_only", "content_band", "anchor_preservation", "random_stratified"]
    models = [m for m in ("model_1", "model_2") if m in df["model_id"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = range(len(conds))
    width = 0.25

    for ax, m in zip(axes, models):
        sub = df[df["model_id"] == m]
        acc = [sub[sub["condition"] == c]["token_accuracy"].mean() * 100 for c in conds]
        hal = [sub[sub["condition"] == c]["hallucination_rate"].mean() * 100 for c in conds]
        omi = [sub[sub["condition"] == c]["omission_rate"].mean() * 100 for c in conds]

        ax.bar([i - width for i in x], acc, width, label="Corretti (Accuracy)", color="#54A24B")
        ax.bar(list(x), hal, width, label="Allucinazioni (su pred.)", color="#E45756")
        ax.bar([i + width for i in x], omi, width, label="Omissioni (su oracle)", color="#4C78A8")

        ax.set_title(f"{MODEL_LABELS.get(m, m)} (media A+B, 10%–50%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([COND_LABELS[c] for c in conds])
        ax.set_ylim(0, 108)
        ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("Percentuale (%)")

    fig.suptitle("Profilo degli errori per condizione di selezione lessicale", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_errori_per_condizione.png")
    plt.close(fig)


def plot_controls(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    conds = ["function_only", "content_band", "anchor_preservation", "random_stratified"]
    base_df = df[df["model_id"] == df["model_id"].unique()[0]]
    x = range(len(conds))
    func_vals = [base_df[base_df["condition"] == c]["elided_function_word_ratio"].mean() * 100 for c in conds]
    cont_vals = [base_df[base_df["condition"] == c]["elided_content_ratio"].mean() * 100 for c in conds]
    zipf_vals = [base_df[base_df["condition"] == c]["elided_mean_zipf_rank"].mean() for c in conds]

    ax1 = axes[0]
    ax1.bar(x, func_vals, 0.55, label="Function words (%)", color="#4C78A8")
    ax1.bar(x, cont_vals, 0.55, bottom=func_vals, label="Content words (%)", color="#F58518")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([COND_LABELS[c] for c in conds])
    ax1.set_ylabel("Composizione parole elise (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title("Controllo elided_composition per condizione")
    for i, z in enumerate(zipf_vals):
        ax1.text(i, 103, f"Zipf medio:\n{z:.0f}", ha="center", va="bottom", fontsize=8)
    ax1.legend(fontsize=8.5, loc="lower right")

    ax2 = axes[1]
    for m, color in (("model_1", "#F58518"), ("model_2", "#54A24B")):
        gap_csv = PHASE7_DIR / m / "gap_context_accuracy.csv"
        if not gap_csv.exists():
            continue
        gdf = pd.read_csv(gap_csv)
        gdf["density_bin"] = pd.qcut(gdf["context_density"], q=6, duplicates="drop")
        binned = gdf.groupby("density_bin", observed=True).agg(
            mean_d=("context_density", "mean"),
            mean_acc=("gap_accuracy", "mean"),
        )
        r, p = spearmanr(gdf["context_density"], gdf["gap_accuracy"])
        ax2.plot(
            binned["mean_d"] * 100,
            binned["mean_acc"] * 100,
            marker="o",
            linewidth=2,
            color=color,
            label=f"{MODEL_LABELS.get(m, m)} (ρ={r:.3f}, p={p:.1e})",
        )

    ax2.set_title("Controllo context_density (±50 parole) vs Accuratezza gap")
    ax2.set_xlabel("Densità media di anchor words nel contesto residuo (%)")
    ax2.set_ylabel("Accuratezza media per gap (%)")
    ax2.legend(fontsize=8.5)

    fig.suptitle("Controlli sperimentali del Protocollo v1.1", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_controlli_protocollo.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    combined_csv = PHASE7_DIR / "combined_summary.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"Esegui prima experiments/run_phase7.py ({combined_csv} mancante)")
    df = pd.read_csv(combined_csv)

    plot_compression_vs_baseline(df)
    plot_fidelity_by_ratio(df)
    plot_errors_by_condition(df)
    plot_controls(df)
    print(f"Grafici salvati in: {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
