## 6. Changelog

### v1.1
- Aggiunto secondo fattore sperimentale `word_selection` con 4 condizioni:
  random_stratified (baseline), function_only (ablation), content_band (primary),
  anchor_preservation (primary).
- Rilassata la contiguità dei gap (`contiguous: false`) per consentire
  selezione selettiva delle parole da elidere.
- Aggiunte diagnostiche `context_density` e `elided_composition` in `metrics.yaml`.
- Completati `base.yaml:paths/scales/splits` e `tokenizer.yaml:placeholder`.
