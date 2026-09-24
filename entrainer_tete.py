"""Entraîne la tête de projection servie en production, à partir du catalogue.

Étape 3 du pipeline : construire_catalogue.py produit data/catalogue.npz (embeddings bruts du
backbone, un label par monture), ce script entraîne par-dessus une petite tête qui rapproche
les vues d'une même monture et éloigne les autres, puis l'écrit dans data/tete.pt. app.py la
charge automatiquement si elle existe.

Le découpage train/val se fait PAR MONTURE : la validation ne contient que des montures jamais
vues à l'entraînement, seule façon de mesurer ce qui se transfère à une monture nouvellement
entrée en stock.

Pré-entraînement facultatif : si data/kaggle_preentrainement.npz existe, la tête apprend
d'abord à discriminer les ~1800 modèles du jeu Kaggle, puis se spécialise sur le stock réel à
taux d'apprentissage réduit. Mesuré : +0.063 de recall@1 sur des montures jamais vues, le stock
réel seul (78 montures d'entraînement) étant trop petit pour cette tâche.

Usage :
    python entrainer_tete.py
    python entrainer_tete.py --sans-preentrainement
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from extraire_embeddings import assigner_split
from finetune_triplet import _projeter, entrainer, etalement
from recall_clip import KS, evaluer_recall
from recall_grid import BACKBONE_PROD


def verifier_preentrainement(backbone: str | None, dim: int, dim_catalogue: int, chemin) -> None:
    """Refuse un fichier de pré-entraînement qui n'a pas été encodé avec le backbone de
    production. On compare le NOM du backbone, pas seulement la dimension : deux backbones
    différents de même dimension passeraient le test dimensionnel tout en produisant des espaces
    incompatibles, et le pré-entraînement dégraderait au lieu d'aider, sans rien signaler."""
    if backbone != BACKBONE_PROD or dim != dim_catalogue:
        raise RuntimeError(
            f"{chemin} : backbone '{backbone}' en dimension {dim}, alors que la production "
            f"utilise '{BACKBONE_PROD}' en dimension {dim_catalogue}. Le ré-encoder, ou lancer "
            f"--sans-preentrainement."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--catalogue", type=Path, default=Path("data/catalogue.npz"))
    p.add_argument("--sortie", type=Path, default=Path("data/tete.pt"))
    p.add_argument("--preentrainement", type=Path, default=Path("data/kaggle_preentrainement.npz"))
    p.add_argument("--sans-preentrainement", action="store_true")
    p.add_argument("--dim-sortie", type=int, default=128)
    p.add_argument("--pas", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-specialisation", type=float, default=1e-4,
                    help="taux réduit pour la phase sur stock réel, après pré-entraînement")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.catalogue, allow_pickle=True)
    emb, labels = d["emb"], d["labels"].astype(str)
    print(f"catalogue : {len(emb)} photos, {len(set(labels))} montures, "
          f"backbone {str(d['backbone']) if 'backbone' in d else '(non précisé)'}")

    # val/test fusionnés : on ne fait pas de sélection d'hyperparamètres ici, donc un seul
    # jeu de montures jamais vues suffit pour l'arrêt anticipé et le chiffre annoncé.
    assign = assigner_split(labels, seed=args.seed)
    est_train = np.array([assign[l] == "train" for l in labels])
    emb_train, labels_train = emb[est_train], labels[est_train]
    emb_val, labels_val = emb[~est_train], labels[~est_train]
    print(f"  train {len(set(labels_train))} montures | validation {len(set(labels_val))} montures "
          f"(jamais vues à l'entraînement)")

    avant = evaluer_recall(emb_val, labels_val)["recall"]

    tete_initiale, lr = None, args.lr
    if args.preentrainement.exists() and not args.sans_preentrainement:
        kg = np.load(args.preentrainement, allow_pickle=True)
        emb_k, labels_k = kg["emb"], kg["labels"].astype(str)
        backbone_kg = str(kg["backbone"]) if "backbone" in kg else None
        verifier_preentrainement(backbone_kg, emb_k.shape[1], emb.shape[1], args.preentrainement)
        print(f"\nPré-entraînement sur {len(emb_k)} images, {len(set(labels_k))} classes "
              f"({args.preentrainement.name}) :")
        assign_k = assigner_split(labels_k, seed=args.seed)
        est_train_k = np.array([assign_k[l] == "train" for l in labels_k])
        tete_initiale, _ = entrainer(
            emb_k[est_train_k], labels_k[est_train_k], emb_k[~est_train_k], labels_k[~est_train_k],
            dim_sortie=args.dim_sortie, p=16, k=3, pas=args.pas, lr=args.lr,
            device="cpu", seed=args.seed, verbose=True,
        )
        lr = args.lr_specialisation
        print(f"\nSpécialisation sur le stock réel (lr réduit à {lr}) :")

    tete, _ = entrainer(
        emb_train, labels_train, emb_val, labels_val,
        dim_sortie=args.dim_sortie, p=16, k=3, pas=args.pas, lr=lr,
        device="cpu", seed=args.seed, verbose=True, tete_initiale=tete_initiale,
    )
    projete = _projeter(tete, emb_val, "cpu")
    apres = evaluer_recall(projete, labels_val)["recall"]

    print("\nSur les montures de validation (jamais vues) :")
    for k in KS:
        delta = apres[k] - avant[k]
        print(f"  recall@{k:<2} : {avant[k]:.3f} -> {apres[k]:.3f}  ({delta:+.3f})")
    print(f"  étalement des embeddings : {etalement(projete):.3f}")

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tete.state_dict(), args.sortie)
    print(f"\nTête sauvegardée : {args.sortie}")
    print("Redémarre l'API (uvicorn) pour qu'elle la charge.")


if __name__ == "__main__":
    main()
