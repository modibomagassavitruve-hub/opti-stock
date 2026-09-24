"""Parsing du texte OCR d'une étiquette de monture -> champs structurés.

Exemples d'étiquettes (intérieur de branche) :
    "RAY-BAN RB3025 001/51 58□14 135"
    "Persol PO3007V 95 52-20 145"

Stdlib uniquement : testable sans GPU ni dépendance externe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Séparateur toléré entre calibre / pont / branche : le symbole □ est souvent
# lu comme [], -, O, ou pas du tout par l'OCR.
_SEP = r"\s*[^\d\s]{0,2}\s*"
# Plages réalistes : calibre 40-65, pont 10-24, branche 120-159.
# "0?" gère le cas où le carré □ est lu comme un zéro ("52018 140").
RE_TAILLE = re.compile(
    rf"(?<!\d)(4\d|5\d|6[0-5]){_SEP}0?(1\d|2[0-4]){_SEP}(1[2-5]\d)(?!\d)"
)
RE_REFERENCE = re.compile(r"^[A-Za-z]{1,4}[-.]?\d{3,5}[A-Za-z]{0,3}$")
RE_REFERENCE_NUM = re.compile(r"^\d{4,5}$")
RE_COLORIS = re.compile(r"^(?:\d{2,4}|[A-Za-z]{1,2}\d{1,3})(?:/[A-Za-z0-9]{1,3})?$")


@dataclass
class Etiquette:
    marque: str = ""
    reference: str = ""
    coloris_code: str = ""
    calibre: str = ""
    pont: str = ""
    branche: str = ""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _tokens(texte: str) -> list[str]:
    bruts = (t.strip(",;:|()") for t in re.split(r"\s+", texte))
    return [t for t in bruts if t]


def _detecter_marque(tokens: list[str], marques: list[str]) -> str:
    """Meilleure correspondance approchée avec la liste de marques connues."""
    if not marques:
        return ""
    table = {_norm(m): m for m in marques}
    candidats = [t for t in tokens if len(_norm(t)) >= 3]
    candidats += [a + b for a, b in zip(tokens, tokens[1:])]  # marques en 2 mots
    meilleur, score = "", 0.0
    for c in candidats:
        cn = _norm(c)
        for mn, m in table.items():
            r = SequenceMatcher(None, cn, mn).ratio()
            if r > score:
                meilleur, score = m, r
    return meilleur if score >= 0.85 else ""


def _detecter_reference(tokens: list[str]) -> tuple[str, list[str]]:
    """Retourne (référence, tokens restants sans ceux qui la composent)."""
    for i, t in enumerate(tokens):
        if RE_REFERENCE.match(t):
            return t, tokens[:i] + tokens[i + 1:]
    # Référence écrite en deux morceaux : "RB 3025"
    for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
        if re.fullmatch(r"[A-Za-z]{1,4}", a) and re.fullmatch(r"\d{3,5}[A-Za-z]{0,3}", b):
            return a + b, tokens[:i] + tokens[i + 2:]
    for i, t in enumerate(tokens):
        if RE_REFERENCE_NUM.match(t):
            return t, tokens[:i] + tokens[i + 1:]
    return "", tokens


def _detecter_coloris(tokens: list[str]) -> str:
    for t in tokens:
        if not any(ch.isdigit() for ch in t):
            continue
        # Correction classique : O lu à la place de 0 ("OO1/51")
        corrige = t.replace("O", "0").replace("o", "0")
        if RE_COLORIS.match(corrige):
            return corrige
    return ""


def parse_etiquette(texte: str, marques: list[str] | None = None) -> Etiquette:
    """Extrait marque, référence, coloris et calibre-pont-branche d'un texte OCR."""
    res = Etiquette()
    reste = texte

    # La taille est en général à la fin de l'étiquette : on garde le dernier match.
    matches = list(RE_TAILLE.finditer(texte))
    if matches:
        m = matches[-1]
        res.calibre, res.pont, res.branche = m.group(1), m.group(2), m.group(3)
        reste = texte[: m.start()] + " " + texte[m.end():]

    tokens = _tokens(reste)
    res.marque = _detecter_marque(tokens, marques or [])
    res.reference, restants = _detecter_reference(tokens)
    res.coloris_code = _detecter_coloris(restants)
    return res
