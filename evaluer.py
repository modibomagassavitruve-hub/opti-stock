"""Mesure la performance de recherche dans les conditions d'usage réel.

Ce protocole est le résultat d'une série de mesures fausses. Trois pièges l'ont rendu
nécessaire, et chacun donne des chiffres flatteurs :

1. L'AUTO-APPARIEMENT. Interroger avec une photo présente au catalogue donne une similarité
   de ~0.99 et un recall proche de 1. Ça ne mesure rien. Ici une photo par monture est retirée
   du catalogue pour servir de requête.

2. LA FUITE PAR L'ENTRAÎNEMENT. Si la tête a été entraînée sur la photo de requête, elle l'a
   mémorisée. Mesuré : recall@1 de 0.77 avec fuite contre 0.26 sans. La tête est donc
   réentraînée ici sans les photos de requête.

3. LE RACCOURCI DU DÉCOR. Deux photos prises à quelques secondes partagent le même fond, et le
   modèle peut reconnaître l'endroit plutôt que la monture. La "marge honnête" compare les
   montures identiques à des montures DIFFÉRENTES photographiées dans la même minute, donc à
   décor équivalent. Sans ce contrôle, l'effet du décor a compté jusqu'à six fois le signal
   d'identité.

Usage :
    python evaluer.py
    python evaluer.py --par-taille     # effet du nombre de montures en concurrence
    python evaluer.py --seuils         # couverture et précision selon le seuil de confiance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from finetune_triplet import (TeteProjection, _projeter, echantillonner_batch_pk,
                              encoder_labels, perte_supcon)


def horodatage(chemin: str) -> int:
    """Secondes depuis minuit, lues dans le nom de fichier (20260923_162505.jpg). -1 si absent."""
    try:
        t = Path(chemin).stem.split("_")[1]
        return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    except Exception:
        return -1


def separer_requetes(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Retire une photo par monture : elle servira de requête, le reste fait le catalogue."""
    rng = np.random.default_rng(seed)
    requete = np.zeros(len(labels), bool)
    for m in sorted(set(labels)):
        requete[rng.choice(np.where(labels == m)[0])] = True
    return requete


def entrainer_sans_les_requetes(emb, labels, catalogue, dim_sortie=128, pas=1500, seed=0,
                                 variantes: np.ndarray | None = None):
    """Entraîne la tête sur le seul catalogue. Pas d'arrêt anticipé : aucune photo ne peut être
    réservée pour la validation sans réintroduire une fuite.

    `variantes` (V, N, D) reproduit l'augmentation utilisée en production. Sans elle, la mesure
    ne décrirait pas le modèle réellement servi."""
    torch.manual_seed(seed)
    vues = emb[catalogue][None] if variantes is None else np.concatenate(
        [emb[catalogue][None], variantes[:, catalogue]])
    tete = TeteProjection(emb.shape[1], dim_sortie=dim_sortie)
    opt = torch.optim.Adam(tete.parameters(), lr=1e-3)
    E = torch.tensor(vues, dtype=torch.float32)
    yi = encoder_labels(labels[catalogue])
    rng = np.random.default_rng(seed)
    for _ in range(pas):
        idx = echantillonner_batch_pk(labels[catalogue], 16, 3, rng)
        v = rng.integers(E.shape[0], size=len(idx))
        perte = perte_supcon(tete(E[v, idx]), torch.tensor(yi[idx]))
        opt.zero_grad(); perte.backward(); opt.step()
    return tete


def mesurer(e, labels, ts, requete, catalogue, ks=(1, 5, 10)) -> dict:
    S = e[requete] @ e[catalogue].T
    yq, yc = labels[requete], labels[catalogue]
    ordre = np.argsort(-S, axis=1)
    res = {f"recall@{k}": float(np.mean([yq[i] in yc[ordre[i, :k]] for i in range(len(yq))]))
           for k in ks}
    res["similarite_top1"] = np.take_along_axis(S, ordre[:, :1], axis=1).ravel()
    res["juste_top1"] = np.array([yc[ordre[i, 0]] == yq[i] for i in range(len(yq))])
    res["juste_top5"] = np.array([yq[i] in yc[ordre[i, :5]] for i in range(len(yq))])

    # marge honnête : montures différentes photographiées dans la même minute = même décor
    plein = e @ e.T
    hd = ~np.eye(len(labels), dtype=bool)
    dt = np.abs(ts[:, None] - ts[None, :])
    memes = (labels[:, None] == labels[None, :]) & hd
    voisines = (labels[:, None] != labels[None, :]) & (dt >= 20) & (dt <= 150) & (ts[:, None] >= 0)
    res["marge_honnete"] = (float(plein[memes].mean() - plein[voisines].mean())
                            if voisines.any() else float("nan"))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--catalogue", type=Path, default=Path("data/catalogue.npz"))
    p.add_argument("--par-taille", action="store_true")
    p.add_argument("--seuils", action="store_true")
    p.add_argument("--variantes", type=int, default=4,
                    help="reproduit l'augmentation de entrainer_tete.py (0 pour l'ignorer)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.catalogue, allow_pickle=True)
    emb, labels = d["emb"], d["labels"].astype(str)
    ts = np.array([horodatage(c) for c in d["chemins"].astype(str)])
    marques = d["marques"].astype(str) if "marques" in d else None

    requete = separer_requetes(labels, args.seed)
    catalogue = ~requete
    print(f"{len(set(labels))} montures | {catalogue.sum()} photos au catalogue | "
          f"{requete.sum()} requêtes, jamais vues à l'entraînement")

    variantes = None
    if args.variantes and "chemins" in d:
        from entrainer_tete import encoder_variantes
        print(f"encodage de {args.variantes} variantes par photo (comme à l'entraînement)…")
        variantes = encoder_variantes(d["chemins"].astype(str), args.variantes, seed=args.seed)

    tete = entrainer_sans_les_requetes(emb, labels, catalogue, seed=args.seed, variantes=variantes)
    e = _projeter(tete, emb, "cpu")

    brut = mesurer(emb, labels, ts, requete, catalogue)
    avec = mesurer(e, labels, ts, requete, catalogue)
    print(f"\n{'':22} {'backbone brut':>14} {'+ tête':>10}")
    for k in ("recall@1", "recall@5", "recall@10", "marge_honnete"):
        print(f"  {k:20} {brut[k]:>14.3f} {avec[k]:>10.3f}")

    if marques is not None and (marques != "").any():
        connues = requete & (marques != "")
        sans, filtre = [], []
        for i in np.where(connues)[0]:
            for cible, masque in ((sans, catalogue), (filtre, catalogue & (marques == marques[i]))):
                sims = e[masque] @ e[i]
                cible.append(labels[i] in labels[masque][np.argsort(-sims)[:5]])
        print(f"\n  filtre par marque ({connues.sum()} requêtes dont la marque est connue) :")
        print(f"    recall@5 sans filtre : {np.mean(sans):.3f}")
        print(f"    recall@5 avec filtre : {np.mean(filtre):.3f}")

    if args.par_taille:
        print("\neffet du nombre de montures en concurrence :")
        montures = sorted(set(labels))
        rng = np.random.default_rng(args.seed)
        for n in (20, 40, 60, 80, len(montures)):
            if n > len(montures):
                continue
            scores = []
            for _ in range(8):
                ech = rng.choice(montures, size=n, replace=False)
                m = np.isin(labels, ech)
                scores.append(mesurer(e, labels, ts, requete & m, catalogue & m)["recall@5"])
            print(f"  {n:4d} montures : recall@5 = {np.mean(scores):.3f}")

    if args.seuils:
        print("\ncouverture et précision selon le seuil de confiance :")
        print(f"  {'seuil':<8}{'couverture':<16}{'précision@1':<14}précision@5")
        for s in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            g = avec["similarite_top1"] >= s
            if not g.any():
                print(f"  {s:<8.2f}{'0%':<16}{'-':<14}-"); continue
            print(f"  {s:<8.2f}{f'{100 * g.mean():.0f}% ({g.sum()})':<16}"
                  f"{avec['juste_top1'][g].mean():<14.3f}{avec['juste_top5'][g].mean():.3f}")


if __name__ == "__main__":
    main()
