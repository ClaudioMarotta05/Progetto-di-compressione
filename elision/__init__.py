"""Package elision: bucatura deterministica del testo.

Moduli:
- seed: derivazione SHA-256 del seed per documento
- tokenizer: tokenizzatore con finditer e gestione placeholder
- word_selection: condizioni di selezione delle parole da elidere
- generator: generazione di scheletro, manifest, oracle
"""

from .seed import seed_doc
from .tokenizer import tokenize, tokenize_with_placeholders, Token

__all__ = ["seed_doc", "tokenize", "tokenize_with_placeholders", "Token"]
