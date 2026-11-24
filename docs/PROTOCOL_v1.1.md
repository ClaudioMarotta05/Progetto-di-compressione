# Protocollo Sperimentale v1.1 — Gap Compression

Stato: CONGELATO
Data di congelamento: 2026-11-20 (v1.0), 2026-11-24 (v1.1)
Commit di riferimento Fase 0: vedi `logs/commit_fase0.txt`
Commit di riferimento Fase 1: vedi `logs/commit_fase1.txt`
Sostituisce: `docs/PROTOCOL_v1.0.md` (mantenuto come storico)

## 1. Obiettivo

Ridurre il peso di un testo tramite elisione deterministica di parole
(bucatura), inferenza dei gap, compressione lossless di scheletro e
correzioni, decompressione e ricostruzione verificabile. Ogni fase è
riproducibile bit a bit a parità di configurazione e seed.

## 2. Definizioni congelate

### 2.1 Parola
Vedi `config/tokenizer.yaml`. Una parola è una sequenza massimale di
caratteri che soddisfa `[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+`. I token non-parola
(punteggiatura, spazi, simboli, placeholder) non sono mai candidati
all'elisione.

### 2.2 Gap
Un gap è un insieme di parole selezionate per l'elisione. Se le parole
selezionate sono consecutive, vengono raggruppate in un unico gap. I gap
non si sovrappongono. In v1.1 non c'è limite superiore alla lunghezza del
gap: `max_gap_length_words: 8` è un residuo di v1.0 e non è applicato.

La selezione delle parole da elidere dipende da `word_selection.conditions`
(vedi §2.4). Il campo `contiguous: false` in `config/elision.yaml` indica
che la selezione è per parola, non per span contiguo.

### 2.3 Strategie
- A: elisione pura + manifest, il modello non vede il conteggio.
- B: placeholder `[N]`, il modello vede il conteggio.

Nessuna altra strategia è ammessa nel protocollo v1.1.

La generazione dei gap **non dipende dalla strategia**: A e B condividono
gli stessi gap a parità di `doc_id`, `ratio`, `word_selection_condition` e
`global_seed`. La strategia influisce solo sul rendering dello scheletro
(placeholder sì/no).

### 2.4 Word selection
Il disegno sperimentale è a due fattori: strategia (A vs B) × condizione
di `word_selection` (4 livelli). Le condizioni sono definite in
`config/elision.yaml:gap_generation.word_selection.conditions`:

- `random_stratified` — baseline. Selezione uniforme per blocco, senza
  logica contestuale. Richiede `n_blocks`.
- `function_only` — ablation. Elide solo function words. Richiede
  `data/external/it_stopwords.txt`.
- `content_band` — primary. Elide parole di contenuto in una fascia di
  frequenza intermedia (default rank 500–5000). Richiede
  `data/external/it_frequency_list.txt`.
- `anchor_preservation` — primary. Preserva le top_k parole per IDF in
  ogni finestra di `window_words` parole, elide il resto. Richiede
  `data/external/it_idf.txt`.

Il tetto strutturale di ogni condizione è `|candidates| / W`, calcolato
per documento e registrato in `meta.json` come `candidate_ratio`. Il campo
`target_reachable` in `meta.json` indica se `target <= |candidates|`.

Il campo `rng_call_order: ["word", "L", "offset"]` in `config/elision.yaml`
è descrittivo dell'ordine delle chiamate a `rng` in v1.0; in v1.1
l'implementazione in `elision/word_selection.py` non lo enforcea. Il campo
è mantenuto per compatibilità con i test di coerenza.

### 2.5 Rapporti
10%, 20%, 30%, 40%, 50% delle parole del documento. Il rapporto effettivo
è registrato nel manifest come `effective_ratio`.

### 2.6 Seed
Seed globale 42 in `config/base.yaml`. Seed per documento:
`int(sha256(f"{global_seed}:{doc_id}").hexdigest(), 16)`. Mai `hash()` di
Python.

### 2.7 Modelli
Almeno due modelli in `config/inference.yaml`, con revisione bloccata.
Decoding greedy, temperatura 0, `do_sample: false`, seed fissi per Python
e PyTorch.

## 3. Metriche congelate

Vedi `config/metrics.yaml`. In sintesi:

- Fedeltà: BLEU (sacrebleu, tokenizer 13a, corpus-level), ROUGE-L (F1, no
  stemmer, document-level, media), Precision@k (k = 1, 3, 5, 10;
  gap-level exact match). Casi limite: `if_fewer_than_k_gaps` usa
  `k_eff = min(k, n_gaps)`; `if_zero_gaps` restituisce `null` ed è
  escluso dall'aggregato.
- Errori: classi `correct`, `hallucinated`, `omitted` (token-level),
  `wrong_length`, `overinserted` (gap-level). Tassi separati:
  accuratezza per parola, allucinazione, omissione, lunghezza errata,
  inserimento eccessivo.
- Compressione: stream separati (`skeleton`, `manifest_or_placeholder`,
  `corrections`), algoritmi DEFLATE, BWT, PPM. Rapporti per stream,
  combinato, overhead manifest/placeholder, verifica lossless bit a bit.
  Formule esplicite in `config/metrics.yaml:compression.formulas`.
- Ricostruzione: identità lossless, qualità vs oracle, distinzione tra
  errore di compressione ed errore di inferenza, verifica che l'oracle
  non sia nel payload.

### 3.1 Controlli
I controlli in `config/metrics.yaml:controls` non sono metriche di
qualità. Servono a controllare il confondente "quantità di contesto
residuo" e "composizione delle parole elise":

- `context_density`: anchor words in ±`window_words` attorno al gap /
  parole totali nella finestra. Mette in relazione il contesto residuo
  con l'accuratezza per gap.
- `elided_composition`: `content_ratio`, `function_word_ratio`,
  `mean_zipf_rank`. Mostra che le condizioni sperimentali elidono classi
  di parole diverse.

## 4. Vincoli

- L'hold-out (`data/holdout/`) non viene letto da nessuno script fino alla
  Fase 7. Ogni violazione invalida la valutazione finale.
- L'oracle non entra mai nel payload compresso né nel manifest.
- Il manifest non contiene le parole originali (`removed_words`,
  `removed_text`, `oracle_text` vietati).
- Ogni esecuzione registra: hash dei dati, hash delle config, hash del
  commit, seed, versione dei modelli.
- Il determinismo è verificato eseguendo due volte la stessa
  configurazione e confrontando gli output.
- Il protocollo v1.1 sostituisce il v1.0 per tutti gli esperimenti
  eseguiti da questo commit in poi. Il v1.0 resta valido solo per gli
  esperimenti già eseguiti prima di questo commit.

## 5. Modifiche al protocollo

Qualsiasi modifica a `config/elision.yaml`, `config/metrics.yaml`,
`config/inference.yaml`, `config/base.yaml`, `config/tokenizer.yaml` o a
questo documento richiede una nuova versione (`v1.2`, `v1.3`, ...) e un
nuovo commit di riferimento. Le versioni precedenti restano valide per
gli esperimenti già eseguiti.

## Appendice A — Changelog

### v1.1
- Aggiunto secondo fattore sperimentale `word_selection` con 4 condizioni:
  `random_stratified` (baseline), `function_only` (ablation),
  `content_band` (primary), `anchor_preservation` (primary).
- Rilassata la contiguità dei gap (`contiguous: false`) per consentire
  selezione selettiva delle parole da elidere.
- Aggiunte diagnostiche `context_density` e `elided_composition` in
  `config/metrics.yaml:controls`.
- Completati `config/base.yaml:paths/scales/splits` e
  `config/tokenizer.yaml:placeholder`.
- Aggiunti `level` e `aggregation` a BLEU e ROUGE-L, casi limite a
  Precision@k, `errors.levels`, `compression.formulas`.
- Aggiunto `word_selection_condition` al manifest; aggiunto `orig_token_end`;
  aggiunto `removed_text` ai `forbidden_fields`.

### v1.0
- Versione iniziale. Vedi `docs/PROTOCOL_v1.0.md`.