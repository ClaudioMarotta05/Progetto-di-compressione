"""Tokenizzatore deterministico basato su regex.

Una parola è una sequenza massimale che matcha word_pattern.
I token non-parola sono conservati e non sono mai candidati all'elisione.
Il placeholder [N] è un token atomico non-parola.
"""

import re
from dataclasses import dataclass
from typing import List

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
PLACEHOLDER_PATTERN = re.compile(r"\[[0-9]+\]")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    is_word: bool


def tokenize(text: str) -> List[Token]:
    """Tokenizza il testo in token parola e non-parola, coprendo tutto il testo."""
    tokens: List[Token] = []
    pos = 0
    for m in WORD_PATTERN.finditer(text):
        if m.start() > pos:
            # token non-parola tra due parole
            tokens.append(Token(text[pos:m.start()], pos, m.start(), False))
        tokens.append(Token(m.group(), m.start(), m.end(), True))
        pos = m.end()
    if pos < len(text):
        tokens.append(Token(text[pos:], pos, len(text), False))
    return tokens


def tokenize_with_placeholders(text: str) -> List[Token]:
    """Come tokenize, ma riconosce [N] come token atomico non-parola."""
    tokens: List[Token] = []
    pos = 0
    # scansiona placeholder e parole insieme
    combined = re.compile(r"\[[0-9]+\]|[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
    for m in combined.finditer(text):
        if m.start() > pos:
            tokens.append(Token(text[pos:m.start()], pos, m.start(), False))
        is_placeholder = PLACEHOLDER_PATTERN.fullmatch(m.group()) is not None
        tokens.append(Token(m.group(), m.start(), m.end(), not is_placeholder))
        pos = m.end()
    if pos < len(text):
        tokens.append(Token(text[pos:], pos, len(text), False))
    return tokens
