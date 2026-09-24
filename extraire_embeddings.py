"""Extraction des embeddings FashionCLIP (recadrés) pour tout le dataset, avec split train/val/test par produit.

Étape 1 du fine-tuning : ce script ne s'entraîne pas. Il calcule une fois pour toutes les
embeddings FashionCLIP (backbone gelé) de chaque image, après recadrage par Glasses Detector,
et les met en cache avec un split train/val/test assigné par PRODUIT (pas par image), pour
éviter toute fuite entre les jeux (le même modèle de monture ne doit jamais apparaître à la
fois dans le train et dans le test).

Installation (dans l'environnement opti-stock-312, celui avec Python 3.12 et glasses-detector) :
    pip install torch transformers pillow numpy pandas glasses-detector

Usage :
    python extraire_embeddings.py --racine data/kaggle_glasses --profondeur 2

Résultat : un seul fichier data/embeddings_fashionclip.npz avec :
    emb      (N, 512) float32  -- embeddings FashionCLIP normalisés
    labels   (N,)      str      -- identifiant produit de chaque image
    chemins  (N,)      str      -- chemin de l'image (recadrée si --sans-recadrage n'est pas passé)
    splits   (N,)      str      -- 'train' / 'val' / 'test', assigné par produit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recall_clip import calculer_embeddings, lister_images
from recall_grid import recadrer_dossier

MODELE_FASHIONCLIP = "patrickjohncyh/fashion-clip"


def filtrer_couverture(df: pd.DataFrame, max_par_classe: int, min_images: int = 2, seed: int = 0) -> pd.DataFrame:
    """Garde TOUTES les classes d'au moins `min_images` images (pas de plafond sur le nombre
    de classes, contrairement à filtrer() de recall_clip.py, pensé pour un sous-échantillon).
    Plafonne seulement le nombre d'images par classe, pour borner le temps de calcul."""
    tailles = df.groupby("label").size()
    valides = tailles[tailles >= min_images].index
    df = df[df["label"].isin(valides)].sample(frac=1, random_state=seed)  # mélange reproductible
    return (
        df.groupby("label").head(max_par_classe)
        .sort_values(["label", "chemin"])
        .reset_index(drop=True)
    )


def assigner_split(labels: np.ndarray, seed: int = 0, ratios: tuple = (0.7, 0.15, 0.15)) -> dict:
    """Assigne chaque PRODUIT (pas chaque image) à train/val/test : aucun produit ne se
    retrouve dans deux jeux à la fois, pour éviter le data leakage."""
    classes = sorted(set(labels))
    rng = np.random.default_rng(seed)
    ordre = rng.permutation(len(classes))
    n = len(classes)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    assign = {}
    for rang, idx in enumerate(ordre):
        c = classes[idx]
        assign[c] = "train" if rang < n_train else ("val" if rang < n_train + n_val else "test")
    return assign


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/kaggle_glasses"))
    p.add_argument("--profondeur", type=int, default=2)
    p.add_argument("--max-par-classe", type=int, default=15,
                    help="borne le temps de calcul ; augmente si ton temps/disque le permet")
    p.add_argument("--min-images", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sans-recadrage", action="store_true")
    p.add_argument("--sortie", type=Path, default=Path("data/embeddings_fashionclip.npz"))
    args = p.parse_args()

    df = lister_images(args.racine, args.profondeur)
    df = filtrer_couverture(df, args.max_par_classe, args.min_images, args.seed)
    print(f"{len(df)} images, {df['label'].nunique()} classes (produits) retenues")

    chemins = df["chemin"].tolist()
    if not args.sans_recadrage:
        chemins = recadrer_dossier(chemins, args.racine, args.sortie.parent / "crops_full")

    print(f"Calcul des embeddings FashionCLIP ({len(chemins)} images)...")
    emb, gardes = calculer_embeddings(chemins, MODELE_FASHIONCLIP, args.batch)
    labels = df["label"].to_numpy()[gardes].astype(str)
    chemins_gardes = np.array(chemins)[gardes].astype(str)

    assign = assigner_split(labels, args.seed)
    splits = np.array([assign[l] for l in labels])
    for s in ("train", "val", "test"):
        n_img = int((splits == s).sum())
        n_cls = len(set(labels[splits == s]))
        print(f"  {s:5s} : {n_img:5d} images, {n_cls:4d} produits")

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.sortie, emb=emb, labels=labels, chemins=chemins_gardes, splits=splits)
    print(f"\nSauvegardé : {args.sortie}")


if __name__ == "__main__":
    main()
