# Progressive Neural Compression with Lossless Residuals

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Protocol](https://img.shields.io/badge/Protocollo-v1.1%20Frozen-blueviolet)
![Tests](https://img.shields.io/badge/Pytest-19%2F19%20Passed-brightgreen)
![Lossless](https://img.shields.io/badge/Lossless%20Verified-120%2F120%20OK-success)

Pipeline sperimentale e riproducibile per la **compressione neurale progressiva di testo italiano con residui lossless**, sviluppata per il corso di *Compressione Dati* (Università degli Studi di Salerno).

L'idea di base combina l'elisione controllata di parole, la ricostruzione tramite modelli linguistici locali leggeri (LLM) e la codifica lossless dei soli errori residui all'interno di un container binario verificato bit-a-bit (`gapcomp/1`).

---

## Come funziona la pipeline

Il sistema implementa uno schema completo **Encoder → Container → Decoder** che garantisce la ricostruzione identica al testo originale (verifica `SHA-256` bit-a-bit) senza mai includere l'oracle nel payload trasmesso:

1. **Elisione controllata (Fase 2–4):** dato un testo originale, vengono generati deterministicamente uno **scheletro** (testo con parole rimosse), un **manifest** (coordinate strutturali dei gap, privo di oracle) e un **oracle** (parole rimosse, usato solo lato encoder e per la valutazione).
2. **Inferenza neurale (Fase 5):** un LLM locale riceve esclusivamente lo scheletro (elaborato per chunk con greedy decoding deterministico) e predice le parole mancanti.
3. **Calcolo dei residui e impacchettamento (Fase 6 — Encoder):** confrontando la predizione del modello con l'oracle si genera lo stream compatto `corrections` (`=` in caso di exact match, `+testo` solo in caso di errore). I tre stream congelati (`skeleton`, `manifest_or_placeholder`, `corrections`) vengono compressi separatamente con `deflate`, `bwt` e `ppm` nel container binario **`gapcomp/1`**.
4. **Decompressione e verifica lossless (Fase 6 — Decoder):** il decoder apre il container `.gapcomp`, verifica gli `SHA-256` di tutti gli stream, combina scheletro, predizioni del modello e residui (senza accedere all'oracle) e verifica che il testo ricostruito sia identico bit-a-bit all'originale.
5. **Valutazione e controlli (Fase 7):** calcolo delle metriche di fedeltà (`sacrebleu`, `ROUGE-L`, `Precision@k`), classificazione degli errori (`deterministic_subsequence`) e controlli sperimentali (`context_density`, `elided_composition`).

### Strategie sperimentali confrontate

| Strategia | Descrizione | Segnaposto nello scheletro | Informazione visibile al modello |
| :--- | :--- | :---: | :--- |
| **Strategia A** | Elisione pura + manifest | Nessuno | Il modello deve individuare dove mancano parole e inferirle |
| **Strategia B** | Elisione con segnaposto atomici + manifest | `[N]` | Il modello vede esattamente la posizione e il numero `N` di parole rimosse |

Ogni strategia viene testata su **5 rapporti di elisione** (`10%`, `20%`, `30%`, `40%`, `50%`) × **4 condizioni di selezione lessicale** definite in `config/elision.yaml` = **40 combinazioni per documento e modello**.

---

## Stato di avanzamento

| Fase | Obiettivo | Stato | Output / Verifica |
| :--- | :--- | :---: | :--- |
| **Fase 0** | Setup repository, ambiente Python 3.12, gestione dipendenze | ✅ Completata | Venv isolato, `.gitignore`, struttura progetto |
| **Fase 1** | Definizione e congelamento del **Protocollo v1.1** | ✅ Completata | `config/*.yaml` + 11 test di coerenza |
| **Fase 2** | Motore di elisione deterministico (`elision/`) | ✅ Completata | Tokenizer reversibile, segnaposto `[N]` atomici, 5 test unitari |
| **Fase 3** | Preparazione corpora e split (`data/dev/`) con verifica hash | ✅ Completata | 30 documenti sincronizzati via hash tra macOS e Linux |
| **Fase 4** | Generazione scheletri, manifest e oracle (`run_phase2.py`) | ✅ Completata | 160 file generati per `finewiki_it_small_001` (40 combo × 4 file) |
| **Fase 5** | Inferenza LLM deterministica (`inference/`, `run_phase5.py`) | ⏳ Parziale | `model_1` (`0.5B`): **40/40 OK** — `model_2` (`1.5B`): in coda |
| **Fase 6** | Residui lossless, codec `deflate`/`bwt`/`ppm`, container `gapcomp/1` | ✅ Completata | **120/120 verifiche lossless OK** su `model_1`, 19/19 test pytest |
| **Fase 7** | Metriche di qualità (`BLEU`, `ROUGE-L`, `P@k`), errori e controlli | 🛠️ Prossima | Specifiche congelate in `config/metrics.yaml` |

---

## Struttura del repository

```text
.
├── config/                  # Configurazioni congelate del Protocollo v1.1
│   ├── base.yaml            # Path, seed globale (42), split e scale
│   ├── elision.yaml         # Strategie A/B, ratio 10%-50%, 4 condizioni lessicali
│   ├── inference.yaml       # Modelli (0.5B / 1.5B), revisioni pinned, decoding greedy
│   └── metrics.yaml         # Formule congelate (fidelity, errors, compression, controls)
├── elision/                 # Fase 2: tokenizzazione reversibile ed elisione deterministica
│   ├── tokenizer.py         # Tokenizzazione esatta + gestione atomica di [N]
│   ├── seed.py              # Derivazione SHA-256 deterministica del seed
│   ├── word_selection.py    # Classificazione anchor / content / function words
│   └── generator.py         # Generazione skeleton, manifest (no-oracle) e oracle
├── inference/               # Fase 5: inferenza chunked con modelli locali
│   ├── prompts.py           # Template zero-shot puliti per Strategia A e B
│   └── runner.py            # Loader, greedy decoding, prompt lookup e allineamento gap
├── compression/             # Fase 6: compressione e ricostruzione lossless
│   ├── codecs.py            # Wrapper deflate (zlib-9), bwt (bz2-9), ppm (pyppmd)
│   ├── residuals.py         # Codifica/decodifica residui (= / +testo) e ricostruzione A/B
│   └── container.py         # Formato binario gapcomp/1 + verifiche SHA-256 e anti-oracle
├── experiments/             # Script eseguibili della pipeline
│   ├── run_phase2.py        # Generazione dei 40 scheletri + manifest + oracle
│   ├── run_phase5.py        # Esecuzione inferenza per modello (--model, --tags)
│   └── run_phase6.py        # Compressione 3 stream, container .gapcomp e verifica lossless
├── tests/                   # Suite di test unitari (pytest)
│   ├── test_config_consistency.py
│   ├── test_fase2.py
│   └── test_fase6.py
└── logs/                    # Hash di riferimento dei commit per tracciabilità delle fasi
```

> **Nota su Git:** i documenti sorgenti (`data/`) e gli output generati (`experiments/outputs/`) sono esclusi da `.gitignore` per mantenere il repository leggero; la riproducibilità tra macchine diverse è garantita da seed deterministici e hash di controllo.

---

## Setup e installazione

### macOS (Apple Silicon)

```bash
git clone <URL_DEL_REPO>
cd "Progressive Neural Compression with Loseless Residuals"

python3.12 -m venv ~/venvs/gap-compression
source ~/venvs/gap-compression/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux (CPU o GPU NVIDIA CUDA)

```bash
git clone <URL_DEL_REPO>
cd "Progressive Neural Compression with Loseless Residuals"

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Esecuzione passo-passo

### 1. Verifica dei test unitari

```bash
python -m pytest tests/ -v
```

Output atteso: **`19 passed`** (coerenza configurazioni, determinismo elisione, codec, residui, container `gapcomp/1` e controllo anti-oracle).

### 2. Fase 4 — Generazione di scheletri, manifest e oracle

```bash
python experiments/run_phase2.py
```

Genera le 40 combinazioni in `experiments/outputs/phase2/finewiki_it_small_001/`.

### 3. Fase 5 — Inferenza neurale

Modelli configurati in `config/inference.yaml`:
- **`model_1`:** `Qwen/Qwen2.5-0.5B-Instruct` (revisione bloccata `7ae557604adf67be50417f59c2c2f167def9a775`)
- **`model_2`:** `Qwen/Qwen2.5-1.5B-Instruct`

Su **macOS** (usa `caffeinate -i` per evitare lo sleep durante le esecuzioni lunghe):

```bash
caffeinate -i python experiments/run_phase5.py --model model_1
caffeinate -i python experiments/run_phase5.py --model model_2
```

Su **Linux** (rileva automaticamente CUDA se disponibile):

```bash
python experiments/run_phase5.py --model model_2
```

Lo script è **riprendibile**: se interrotto con `Ctrl+C`, al riavvio salta automaticamente le combinazioni già salvate. È anche possibile eseguire una singola combinazione di test con `--tags A_10_anchor_preservation`.

### 4. Fase 6 — Compressione lossless e verifica bit-a-bit

```bash
python experiments/run_phase6.py --model model_1
```

Per elaborare tutti i modelli presenti in `experiments/outputs/phase5/`:

```bash
python experiments/run_phase6.py --model all
```

Produce per ciascun modello:
- `experiments/outputs/phase6/<model_id>/containers/*.gapcomp` (container binari)
- `experiments/outputs/phase6/<model_id>/reports/*.json` (report dettagliati per combinazione)
- `experiments/outputs/phase6/<model_id>/summary.csv` (tabella aggregata per l'analisi)

---

## Scelte tecniche e ottimizzazioni (Fase 5)

Durante il collaudo locale su **MacBook Air M4** sono emersi e stati risolti alcuni aspetti chiave documentati nel codice:

- **CPU `float32` preferita a MPS su Apple Silicon:** il backend `mps` in `float16` ha mostrato instabilità numerica e decadimento dell'output; `cpu` in `float32` garantisce stabilità completa (~26,1 tok/s base).
- **Accelerazione lossless dell'inferenza (`prompt_lookup_num_tokens=10`):** sfruttando il fatto che lo scheletro va in gran parte ricopiato fedelmente, il prompt-lookup decoding di Hugging Face mantiene l'esatto output del greedy decoding (`do_sample=False`, `num_beams=1`) portando il throughput fino a **~110,9 tok/s** (~3,5 ore totali per 40 combinazioni con `model_1`).
- **Chunking deterministico:** lo scheletro viene elaborato in chunk da ~200 parole per evitare saturazione di contesto e troncamenti, poi ricomposto nell'ordine originale.
- **Prompt zero-shot puliti:** i test con prompt 1-shot su modelli da `0.5B` causavano copia dell'esempio o deriva testuale; i template zero-shot in `inference/prompts.py` preservano la fedeltà allo scheletro.
- **Determinismo cross-hardware:** la ricostruzione lossless richiede che encoder e decoder ottengano la medesima predizione $\hat{Y}$. Differenze di backend o precisione (es. `cpu/float32` su ARM64 vs `cuda/float16` su x86_64) possono produrre divergenze sui logit in virgola mobile; pertanto ogni modello completa il ciclo compressione → decompressione sullo stesso ambiente hardware.

---

## Risultati attuali — Fase 6 (`model_1`: `Qwen2.5-0.5B-Instruct`)

Documento di riferimento: **`finewiki_it_small_001`** (`117.304` byte grezzi).

### Baseline di compressione diretta sul testo originale

| Codec | Byte compressi | Rapporto (`comp / orig`) | Verifica lossless |
| :--- | :---: | :---: | :---: |
| `deflate` (`zlib`, level 9) | 30.201 B | `0.257` | ✅ OK |
| `bwt` (`bz2`, level 9) | 26.426 B | `0.225` | ✅ OK |
| `ppm` (`pyppmd`) | 23.722 B | `0.202` | ✅ OK |

### Risultati medi della pipeline neurale + residui (`model_1`, 40 combinazioni)

| Strategia | Codec | Exact Hit Rate | `ratio_combined` | `vs_baseline` | Overhead Manifest (`raw_man / raw_skel`) | Lossless (120/120) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | `ppm` | 0.33% | **0.3488** | `1.725x` | 596.49% | ✅ OK |
| **A** | `bwt` | 0.33% | 0.3973 | `1.764x` | 596.49% | ✅ OK |
| **A** | `deflate` | 0.33% | 0.4862 | `1.888x` | 596.49% | ✅ OK |
| **B** | `ppm` | 0.41% | **0.3634** | `1.797x` | 560.04% | ✅ OK |
| **B** | `bwt` | 0.41% | 0.4113 | `1.826x` | 560.04% | ✅ OK |
| **B** | `deflate` | 0.41% | 0.5061 | `1.966x` | 560.04% | ✅ OK |

**Prime evidenze sperimentali:**
- **Integrità 100% garantita:** tutte le **120/120** verifiche lossless (40 combinazioni × 3 codec) ricostruiscono i `117.304` byte originali bit-a-bit senza accesso all'oracle.
- **Gerarchia dei codec confermata:** `ppm` < `bwt` < `deflate` in tutte le configurazioni.
- **Perché `vs_baseline` > 1x con `0.5B`:** emergono due fattori netti dai dati: (1) un modello da mezzo miliardo di parametri raggiunge uno `0.33%–0.41%` di exact match sui gap (lasciando molti `[N]` invariati in Strategia B), rendendo lo stream residuo quasi interamente correttivo; (2) la serializzazione JSONL del manifest introduce un overhead strutturale rilevante rispetto allo scheletro. Il confronto con `model_2` (`1.5B`) permetterà di quantificare l'impatto della scala del modello sul trade-off tra qualità predittiva e dimensione dei residui.

---

## Prossimi passi

1. Completare Fase 5 e Fase 6 per **`model_2` (`Qwen/Qwen2.5-1.5B-Instruct`)** per il confronto multi-modello.
2. Implementare la **Fase 7 (`evaluation/` + `experiments/run_phase7.py`)**:
   - Metriche di fedeltà (`sacrebleu`, `ROUGE-L`, `Precision@k` per $k \in \{1, 3, 5, 10\}$)
   - Analisi fine-grained degli errori (`correct`, `hallucinated`, `omitted`, `wrong_length`, `overinserted`)
   - Controlli `context_density` (±50 parole) ed `elided_composition` (`content_ratio`, `function_word_ratio`, `mean_zipf_rank`)
