"""Baseline OCR : photos d'étiquettes -> champs structurés -> évaluation.

Installation :
    pip install easyocr pandas

Arborescence attendue :
    data/raw/<monture_id>/etiquette.jpg     (photo macro de l'intérieur de la branche)
    data/montures.csv                       (vérité terrain, une ligne par monture)

Colonnes de montures.csv :
    monture_id, marque, reference, coloris_code, calibre, pont, branche

Usage :
    python baseline_ocr.py --photos data/raw --verite data/montures.csv --sortie data/ocr_baseline.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from parse_etiquette import parse_etiquette

CHAMPS = ["marque", "reference", "coloris_code", "calibre", "pont", "branche"]


def lire_texte(reader, chemin: Path) -> str:
    """OCR d'une image -> texte sur une seule ligne."""
    return " ".join(reader.readtext(str(chemin), detail=0, paragraph=False))


def _normaliser(valeur: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(valeur).upper())


def evaluer(predictions: pd.DataFrame, verite: pd.DataFrame) -> pd.DataFrame:
    """Compare prédictions et vérité terrain : exact match par champ et global."""
    df = predictions.merge(verite, on="monture_id", suffixes=("_pred", "_vrai"))
    for c in CHAMPS:
        df[f"ok_{c}"] = [
            _normaliser(p) == _normaliser(v)
            for p, v in zip(df[f"{c}_pred"], df[f"{c}_vrai"])
        ]
    df["ok_tout"] = df[[f"ok_{c}" for c in CHAMPS]].all(axis=1)
    return df


def resume(df: pd.DataFrame) -> pd.Series:
    cols = {f"ok_{c}": c for c in CHAMPS} | {"ok_tout": "TOUS LES CHAMPS"}
    return df[list(cols)].mean().rename(index=cols).round(3)


def predire(reader, verite: pd.DataFrame, dossier_photos: Path) -> pd.DataFrame:
    # Baseline : les marques connues viennent de la vérité terrain.
    # Plus tard, elles viendront de la table `marque` du catalogue.
    marques = sorted(set(verite["marque"]) - {""})
    lignes = []
    for monture_id in verite["monture_id"]:
        image = dossier_photos / monture_id / "etiquette.jpg"
        if not image.exists():
            print(f"[absent] {image}")
            continue
        texte = lire_texte(reader, image)
        etiquette = parse_etiquette(texte, marques)
        lignes.append(
            {"monture_id": monture_id, "texte_ocr": texte}
            | {f"{c}": getattr(etiquette, c) for c in CHAMPS}
        )
    return pd.DataFrame(lignes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--photos", default="data/raw", type=Path)
    parser.add_argument("--verite", default="data/montures.csv", type=Path)
    parser.add_argument("--sortie", default="data/ocr_baseline.csv", type=Path)
    args = parser.parse_args()

    import easyocr  # import tardif : le reste du module est testable sans

    reader = easyocr.Reader(["fr", "en"], gpu=False)
    verite = pd.read_csv(args.verite, dtype=str).fillna("")
    predictions = predire(reader, verite, args.photos)
    resultats = evaluer(predictions, verite)
    resultats.to_csv(args.sortie, index=False)

    print("\nExact match par champ (sur", len(resultats), "montures) :")
    print(resume(resultats).to_string())
    echecs = resultats[~resultats["ok_tout"]]
    print(f"\n{len(echecs)} échecs -> ouvre {args.sortie} et lis la colonne texte_ocr pour comprendre les erreurs.")


if __name__ == "__main__":
    main()
