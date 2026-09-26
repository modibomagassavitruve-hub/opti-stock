"""Lecture et écriture de data/montures.csv -- la marque et la référence de chaque monture.

Ces deux champs ne s'extraient pas des photos de façon fiable : mesuré sur 115 montures, l'OCR
n'identifie la marque que dans 30 % des cas, et sur le stock restant il ne lit que les
autocollants de verres (« UV 100% cat.3 ») -- la gravure est à l'intérieur de la branche, qui
n'est pas sur les photos. C'est donc une saisie humaine, et ce module la sert.

L'enjeu justifie la saisie : la marque fait passer le recall@5 de 0.79 à 0.86, et sans elle une
monture ne peut pas être proposée au réseau (aucun confrère ne saurait la reconnaître).

Écriture prudente : le fichier est réécrit en entier à chaque fois, donc on ne touche qu'aux
montures nommées, on refuse celles qu'on ne connaît pas, et on écrit par un fichier temporaire
renommé -- une interruption en cours d'écriture laisserait sinon un CSV tronqué.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

CHAMPS = ("monture", "marque", "reference", "ean", "etat")

# Une marque saisie à la main mérite d'être relue une fois ; une marque déduite de l'OCR ou du
# nom de dossier, à plus forte raison. Ces états disent ce qu'il reste à vérifier.
ETAT_SAISIE = "saisi"
ETATS_A_REVOIR = ("a_saisir", "ocr_a_verifier", "dossier_a_verifier")


def lire(chemin: Path) -> list[dict]:
    if not Path(chemin).is_file():
        return []
    with open(chemin, newline="", encoding="utf-8") as f:
        return [{c: (l.get(c) or "").strip() for c in CHAMPS} for l in csv.DictReader(f)]


def a_completer(chemin: Path) -> list[dict]:
    """Montures dont la marque manque, ou dont la provenance reste à confirmer."""
    return [l for l in lire(chemin)
            if not l["marque"] or l["etat"] in ETATS_A_REVOIR]


def ecrire_marques(chemin: Path, marques: dict[str, str],
                    references: dict[str, str] | None = None,
                    confirmees: set[str] | None = None) -> int:
    """Renseigne la marque (et éventuellement la référence) des montures nommées.

    Une marque vide efface la marque : c'est la façon de corriger une lecture OCR fausse.
    Renvoie le nombre de lignes modifiées. Lève KeyError si une monture est inconnue --
    mieux vaut refuser franchement que d'écrire une ligne fantôme que rien ne relira.

    `confirmees` porte les montures dont un humain a explicitement validé la valeur. Sans
    cela, une ligne renvoyée telle quelle garde son état « à vérifier » : enregistrer une page
    entière ne doit pas transformer une lecture OCR en fait établi, sous peine de promettre au
    réseau une marque que personne n'a relue.
    """
    lignes = lire(chemin)
    if not lignes:
        raise FileNotFoundError(f"{chemin} introuvable ou vide")

    connues = {l["monture"] for l in lignes}
    if inconnues := set(marques) - connues:
        raise KeyError(f"Montures absentes du stock : {', '.join(sorted(inconnues))}")

    references = references or {}
    confirmees = confirmees or set()
    modifiees = 0
    for l in lignes:
        if l["monture"] not in marques:
            continue
        marque = marques[l["monture"]].strip()
        reference = references.get(l["monture"], l["reference"]).strip()

        if not marque:
            etat = "a_saisir"
        elif marque != l["marque"] or l["monture"] in confirmees:
            # Une valeur nouvelle vient forcément d'un humain ; une valeur identique ne vaut
            # confirmation que si elle a été explicitement cochée.
            etat = ETAT_SAISIE
        else:
            etat = l["etat"] or ETAT_SAISIE

        if (marque, reference, etat) == (l["marque"], l["reference"], l["etat"]):
            continue
        l["marque"], l["reference"], l["etat"] = marque, reference, etat
        modifiees += 1

    if modifiees:
        _ecrire(chemin, lignes)
    return modifiees


def _ecrire(chemin: Path, lignes: list[dict]) -> None:
    """Écrit par un temporaire puis renomme : os.replace est atomique, donc une coupure ne peut
    pas laisser un CSV à moitié écrit à la place du stock."""
    chemin = Path(chemin)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                      dir=chemin.parent, delete=False) as f:
        (w := csv.DictWriter(f, CHAMPS)).writeheader()
        w.writerows(lignes)
        temporaire = f.name
    os.replace(temporaire, chemin)


def bilan(chemin: Path) -> dict:
    lignes = lire(chemin)
    return {
        "montures": len(lignes),
        "avec_marque": sum(1 for l in lignes if l["marque"]),
        "sans_marque": sum(1 for l in lignes if not l["marque"]),
        "a_verifier": sum(1 for l in lignes if l["marque"] and l["etat"] in ETATS_A_REVOIR),
        "marques": sorted({l["marque"] for l in lignes if l["marque"]}),
    }
