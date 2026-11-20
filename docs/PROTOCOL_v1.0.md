# Protocollo Sperimentale v1.0 — Gap Compression

Stato: CONGELATO
Data di congelamento: 2026-11-20
Commit di riferimento Fase 0: vedi `logs/commit_fase0.txt`

## 1. Obiettivo

Ridurre il peso di un testo tramite elisione deterministica di parole
(bucatura), inferenza dei gap, compressione lossless di scheletro e
correzioni, decompressione e ricostruzione verificabile. Ogni fase è
riproducibile bit a bit a parità di configurazione e seed.

## 2. Definizioni congelate

### 2.1 Parola
Vedi `config/tokenizer.yaml`. Una parola è una sequenza massimale di
caratteri che soddisfa `[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+`. I token non-parola
(punteggiatura, spazi, simboli) non sono mai candidati all'elisione.

### 2.2 Gap
Uno span contiguo di parole consecutive rimosse. Regole in
`config/elision.yaml:gap_generation`. I gap non si sovrappongono e
non superano 8 parole.

### 2.3 Strategie
- A: elisione pura + manifest, il modello non vede il conteggio.
- B: placeholder `[N]`, il modello vede il conteggio.

Nessuna altra strategia è ammessa nel protocollo v1.0.

### 2.4 Rapporti
10%, 20%, 30%, 40%, 50% delle parole del documento. Il rapporto effettivo
è registrato nel manifest come `effective_ratio`.

### 2.5 Seed
Seed globale 42 in `config/base.yaml`. Seed per documento:
`int(sha256(f"{global_seed}:{doc_id}").hexdigest(), 16)`. Mai `hash()` di Python.

### 2.6 Modelli
Almeno due modelli in `config/inference.yaml`, con revisione bloccata.
Decoding greedy, temperatura 0, `do_sample: false`, seed fissi per Python
e PyTorch.

## 3. Metriche congelate

Vedi `config/metrics.yaml`. In sintesi:

- Fedeltà: BLEU (sacrebleu, tokenizer 13a), ROUGE-L (F1, no stemmer),
  Precision@k (k = 1, 3, 5, 10; gap-level exact match).
- Errori: corretto, allucinato, omesso, lunghezza errata, inserimento
  eccessivo. Tassi separati: accuratezza per parola, allucinazione,
  omissione, lunghezza errata, inserimento eccessivo.
- Compressione: stream separati (scheletro, manifest/placeholder,
  correzioni), algoritmi DEFLATE, BWT, PPM. Rapporti separati e
  combinato, overhead manifest/placeholder, verifica lossless bit a bit.
- Ricostruzione: identità lossless, qualità vs oracle, distinzione
  errore di compressione / errore di inferenza, oracle fuori dal payload.

## 4. Vincoli

- L'hold-out (`data/holdout/`) non viene letto da nessuno script fino alla
  Fase 7. Ogni violazione invalida la valutazione finale.
- L'oracle non entra mai nel payload compresso né nel manifest.
- Il manifest non contiene le parole originali.
- Ogni esecuzione registra: hash dei dati, hash delle config, hash del
  commit, seed, versione dei modelli.
- Il determinismo è verificato eseguendo due volte la stessa
  configurazione e confrontando gli output.

## 5. Modifiche al protocollo

Qualsiasi modifica a `config/elision.yaml`, `config/metrics.yaml`,
`config/inference.yaml` o a questo documento richiede una nuova versione
(`v1.1`, `v1.2`, ...) e un nuovo commit di riferimento. Le versioni
precedenti restano valide per gli esperimenti già eseguiti.
