"""Évalue le pipeline (FashionCLIP + recadrage + tête fine-tunée) sur tes VRAIES photos de montures.

Sert à vérifier le décalage entre les photos e-commerce du dataset Kaggle (fond neutre, studio)
et tes photos réelles de boutique (lumière ambiante, fond quelconque) : c'est le seul moyen
fiable de savoir si le modèle entraîné sur Kaggle généralise à ton usage réel.

Prends au moins 2 photos par monture (idéalement 3-4 : face, 3/4, profil), et range-les ainsi :
    data/mes_montures/<identifiant_monture>/photo1.jpg
    data/mes_montures/<identifiant_monture>/photo2.jpg
    ...
L'identifiant peut être ce que tu veux (référence, nom de dossier), du moment qu'il est unique
par monture et cohérent entre les photos d'une même monture.

Installation : identique aux scripts précédents (déjà en place si tu as suivi les étapes d'avant).

Usage :
    python tester_mes_photos.py --racine data/mes_montures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune_triplet import TeteProjection, evaluer
from recall_clip import KS, calculer_embeddings, evaluer_recall, lister_images
from recall_grid import recadrer_dossier

MODELE_FASHIONCLIP = "patrickjohncyh/fashion-clip"


def filtrer_min_images(df: pd.DataFrame, min_images: int = 2) -> pd.DataFrame:
    """Garde uniquement les montures qui ont au moins `min_images` photos : sans ça, impossible
    de mesurer un recall (il faut au moins un "positif" du même produit à retrouver)."""
    tailles = df.groupby("label").size()
    valides = tailles[tailles >= min_images].index
    return df[df["label"].isin(valides)].reset_index(drop=True)


def construire_detail(chemins: np.ndarray, labels: np.ndarray, res: dict) -> pd.DataFrame:
    """Assemble le tableau détaillé (une ligne par photo, avec sa meilleure correspondance),
    pour repérer concrètement quelles montures posent problème."""
    top1 = res["top_idx"][:, 0]
    return pd.DataFrame({
        "photo": chemins,
        "monture": labels,
        "hit@5": res["hits"][5],
        "top1_photo": chemins[top1],
        "top1_monture": labels[top1],
        "top1_similarite": res["top_sim"][:, 0].round(4),
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/mes_montures"))
    p.add_argument("--tete", type=Path, default=Path("data/tete_finetuned.pt"))
    p.add_argument("--dim-sortie", type=int, default=128)
    p.add_argument("--sans-recadrage", action="store_true")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sortie", type=Path, default=Path("data/mes_montures_resultats.csv"))
    args = p.parse_args()

    df = lister_images(args.racine, profondeur=1)
    df = filtrer_min_images(df)
    if df.empty:
        print(f"Aucune monture avec au moins 2 photos trouvée sous {args.racine}")
        print("Arborescence attendue : data/mes_montures/<id_monture>/photo.jpg (>= 2 par dossier)")
        return
    print(f"{len(df)} photos, {df['label'].nunique()} montures retenues (>= 2 photos chacune)\n")

    chemins = df["chemin"].tolist()
    if not args.sans_recadrage:
        chemins = recadrer_dossier(chemins, args.racine, args.racine.parent / "mes_montures_crops")

    print(f"Calcul des embeddings FashionCLIP ({len(chemins)} photos)...")
    emb, gardes = calculer_embeddings(chemins, MODELE_FASHIONCLIP, args.batch)
    labels = df["label"].to_numpy()[gardes].astype(str)
    chemins_gardes = np.array(chemins)[gardes]

    print("\nBaseline (FashionCLIP gelé, sans fine-tuning), sur tes photos :")
    res_base = evaluer_recall(emb, labels)
    for k in KS:
        print(f"  recall@{k} = {res_base['recall'][k]:.3f}")

    if args.tete.exists():
        device = "cpu"
        tete = TeteProjection(emb.shape[1], dim_sortie=args.dim_sortie)
        tete.load_state_dict(torch.load(args.tete, map_location=device))
        apres = evaluer(tete, emb, labels, device)
        print("\nAprès la tête fine-tunée (entraînée sur Kaggle), sur tes photos :")
        for k in KS:
            delta = apres[k] - res_base["recall"][k]
            signe = "+" if delta >= 0 else ""
            print(f"  recall@{k} = {apres[k]:.3f}  ({signe}{delta:.3f} vs baseline)")
    else:
        print(f"\n[tête fine-tunée introuvable à {args.tete} -> uniquement la baseline évaluée]")

    detail = construire_detail(chemins_gardes, labels, res_base)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.sortie, index=False)
    print(f"\nDétail (baseline) : {args.sortie}  (filtre hit@5 == False pour analyser les échecs)")


if __name__ == "__main__":
    main()
