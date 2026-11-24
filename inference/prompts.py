"""Template di prompt per strategia A e B.

Fissati nel codice per il collaudo minimo (Opzione B).
Dopo il collaudo, spostare in config/prompts.yaml con bump a v1.2.
"""

STRATEGY_A_TEMPLATE = """Sei un assistente che ricostruisce testi italiani.
Ti viene fornito uno scheletro di un testo dove alcune parole sono state
rimosse. I buchi non sono marcati esplicitamente: devi riconoscerli dal
contesto anomalo (per esempio "del nello" indica che una parola è stata
rimossa tra "del" e "nello").

Ricostruisci il testo originale, riempiendo i buchi nel modo più
plausibile. Rispondi SOLO con il testo ricostruito, senza commenti né
spiegazioni.

SCHELETRO:
{skeleton}

TESTO RICOSTRUITO:
"""

STRATEGY_B_TEMPLATE = """Sei un assistente che ricostruisce testi italiani.
Ti viene fornito uno scheletro di un testo dove alcune parole sono state
rimosse. Ogni buco è marcato con un placeholder nella forma [N], dove N è
il numero esatto di parole rimosse in quel punto.

Ricostruisci il testo originale, sostituendo ogni [N] con esattamente N
parole. Rispondi SOLO con il testo ricostruito, senza commenti né
spiegazioni.

SCHELETRO:
{skeleton}

TESTO RICOSTRUITO:
"""


def build_prompt(strategy: str, skeleton: str) -> str:
    if strategy == "A":
        return STRATEGY_A_TEMPLATE.format(skeleton=skeleton)
    if strategy == "B":
        return STRATEGY_B_TEMPLATE.format(skeleton=skeleton)
    raise ValueError(f"Strategia sconosciuta: {strategy!r}. Attese: 'A' o 'B'.")
