"""Template di prompt zero-shot per strategia A e B.
Fissati nel codice per il collaudo minimo (Opzione B).
Dopo il collaudo, spostare in config/prompts.yaml con bump a v1.2.
"""

STRATEGY_A_TEMPLATE = """Sei un correttore di testi italiani. Nel testo seguente alcune parole sono state rimosse senza segnaposto.
Riscrivi l'intero testo inserendo le parole mancanti nei punti in cui la frase è incompleta, mantenendo invariate le parole già presenti.
Rispondi SOLO con il testo ricostruito, senza commenti né spiegazioni.

{skeleton}"""

STRATEGY_B_TEMPLATE = """Sei un correttore di testi italiani. Nel testo seguente alcune parole mancanti sono indicate con segnaposto nella forma [N], dove N è il numero di parole rimosse.
Riscrivi l'intero testo sostituendo ogni [N] con esattamente N parole italiane adatte al contesto, senza lasciare nessuna parentesi quadra [N] nel risultato.
Rispondi SOLO con il testo ricostruito, senza commenti né spiegazioni.

{skeleton}"""


def build_prompt(strategy: str, skeleton: str) -> str:
    if strategy == "A":
        return STRATEGY_A_TEMPLATE.format(skeleton=skeleton)
    if strategy == "B":
        return STRATEGY_B_TEMPLATE.format(skeleton=skeleton)
    raise ValueError(f"Strategia sconosciuta: {strategy!r}. Attese: 'A' o 'B'.")
