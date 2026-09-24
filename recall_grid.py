"""Grille de comparaison : backbones (CLIP, FashionCLIP, DINOv2 cls/mean) x recadrage (Glasses Detector).

Réutilise lister_images / filtrer / evaluer_recall de recall_clip.py : lance-le une fois
seul (--explorer) avant celui-ci pour vérifier que ton dataset est bien lu.

Installation complète (dans l'environnement opti-stock) :
    pip install torch transformers pillow numpy pandas glasses-detector

Chaque condition (backbone x recadrage) est calculée une seule fois et mise en cache sur
disque (data/grille/), donc une grille interrompue peut être relancée sans tout recalculer.

Usage :
    # Grille complète : 4 backbones x {brut, recadré} = 8 conditions
    python recall_grid.py --racine data/kaggle_glasses --profondeur 2

    # Un seul backbone
    python recall_grid.py --racine data/kaggle_glasses --profondeur 2 --backbones dinov2_mean

    # Sans le recadrage Glasses Detector (si son installation pose problème)
    python recall_grid.py --racine data/kaggle_glasses --profondeur 2 --sans-recadrage
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recall_clip import (
    KS,
    _tenseur,
    calculer_embeddings,
    choisir_device,
    evaluer_recall,
    filtrer,
    lister_images,
    ouvrir_image,
)

BACKBONES = {
    "clip": {"famille": "clip", "modele": "openai/clip-vit-base-patch32"},
    "fashionclip": {"famille": "clip", "modele": "patrickjohncyh/fashion-clip"},
    "dinov2": {"famille": "dinov2", "modele": "facebook/dinov2-base", "pooling": "cls"},
    "dinov2_mean": {"famille": "dinov2", "modele": "facebook/dinov2-base", "pooling": "mean"},
    "dinov2_large": {"famille": "dinov2", "modele": "facebook/dinov2-large", "pooling": "cls"},
}

# Backbone servi en production. Mesuré sur 115 montures réelles, images NON recadrées, avec
# tête de projection, moyenne sur 4 découpages (evaluer.py) :
#     fashionclip    recall@5 = 0.841   rec@1 = 0.739   identité = 0.771   512 dimensions
#     dinov2_large   recall@5 = 0.807   rec@1 = 0.713   identité = 0.741   1024 dimensions
#     dinov2         recall@5 = 0.765
#     clip           recall@5 = 0.765
# FashionCLIP gagne sur tout, avec un modèle deux fois plus léger. Le classement s'inverse
# selon qu'on recadre ou non : sur images recadrées dinov2_large l'emportait. FashionCLIP est
# entraîné sur des photos produit e-commerce -- des scènes entières -- là où DINOv2 excelle sur
# des objets détourés ; comparer des backbones sans fixer le prétraitement n'a pas de sens.
# Le catalogue et l'API DOIVENT utiliser le même : construire_catalogue.py l'inscrit dans le
# .npz, charger_modeles refuse un catalogue bâti avec un autre.
BACKBONE_PROD = "fashionclip"


def charger_backbone(nom: str, device: str):
    """Renvoie une fonction (liste d'images PIL -> embeddings normalisés). Implémentation unique
    de l'encodage, partagée par le constructeur de catalogue et par l'API : c'est ce qui garantit
    qu'une photo envoyée à /identifier est encodée exactement comme celles du catalogue."""
    import torch

    conf = BACKBONES[nom]
    famille, nom_modele = conf["famille"], conf["modele"]

    if famille == "clip":
        from transformers import CLIPImageProcessor, CLIPModel

        modele = CLIPModel.from_pretrained(nom_modele).to(device).eval()
        proc = CLIPImageProcessor.from_pretrained(nom_modele)

        def brut(entree):
            return _tenseur(modele.get_image_features(**entree))

    elif famille == "dinov2":
        from transformers import AutoImageProcessor, AutoModel

        modele = AutoModel.from_pretrained(nom_modele).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(nom_modele)
        pooling = conf.get("pooling", "cls")

        def brut(entree):
            h = modele(**entree).last_hidden_state
            return h[:, 0, :] if pooling == "cls" else h[:, 1:, :].mean(dim=1)

    else:
        raise ValueError(f"famille inconnue : {famille}")

    def encoder(images):
        entree = proc(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = torch.nn.functional.normalize(brut(entree), dim=-1)
        return emb.cpu().numpy().astype("float32")

    return encoder


# ---------------------------------------------------------------- embeddings
def _embeddings_dinov2(chemins: list[str], nom_modele: str, pooling: str = "cls", batch: int = 16):
    """Même contrat que calculer_embeddings (recall_clip.py) : renvoie (emb, indices gardés).
    pooling="cls" : token [CLS] de la dernière couche (résumé global appris par le modèle).
    pooling="mean" : moyenne des tokens de patch (hors [CLS]) -> souvent plus fin pour le retrieval."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = choisir_device()
    print(f"Modèle {nom_modele} ({pooling}) sur {device}")
    modele = AutoModel.from_pretrained(nom_modele).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(nom_modele)

    vecteurs, gardes = [], []
    with torch.no_grad():
        for debut in range(0, len(chemins), batch):
            images, idx = [], []
            for j, chemin in enumerate(chemins[debut: debut + batch]):
                try:
                    images.append(ouvrir_image(chemin))
                    idx.append(debut + j)
                except Exception as e:
                    print(f"[ignorée] {chemin}: {e}")
            if not images:
                continue
            entree = proc(images=images, return_tensors="pt").to(device)
            sortie = modele(**entree)
            if pooling == "cls":
                emb = sortie.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                emb = sortie.last_hidden_state[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(f"pooling inconnu : {pooling}")
            emb = torch.nn.functional.normalize(emb, dim=-1)
            vecteurs.append(emb.cpu().numpy().astype("float32"))
            gardes.extend(idx)
            if (debut // batch) % 20 == 0:
                print(f"  {min(debut + batch, len(chemins))}/{len(chemins)}")
    return np.concatenate(vecteurs), np.array(gardes)


def calculer(chemins: list[str], famille: str, nom_modele: str, batch: int, pooling: str | None = None):
    if famille == "clip":
        return calculer_embeddings(chemins, nom_modele, batch)
    if famille == "dinov2":
        return _embeddings_dinov2(chemins, nom_modele, pooling or "cls", batch)
    raise ValueError(f"famille inconnue : {famille}")


# ---------------------------------------------------------------- recadrage
def boite_depuis_masque(masque: np.ndarray, marge: float = 0.08):
    """Boîte englobante (x0, y0, x1, y1) de la monture détectée, avec marge relative.
    None si le masque est vide (rien détecté).

    On ne garde que la plus grande forme du masque : sur une photo prise en boutique, le
    détecteur repère aussi les montures voisines du présentoir, et englober tous les pixels
    donnerait une boîte couvrant toute l'étagère (mesuré : 7 à 13 formes par photo, boîte
    70 à 90 % trop grande). La dilatation préalable recolle les morceaux d'une même monture,
    les montures métal fines ressortant souvent en fragments disjoints."""
    from scipy import ndimage

    binaire = np.asarray(masque) > 0
    if not binaire.any():
        return None

    rayon = max(3, int(0.02 * max(binaire.shape)))
    etiquettes, n = ndimage.label(ndimage.binary_dilation(binaire, iterations=rayon))
    if n > 1:
        # taille mesurée sur le masque d'origine, pas sur sa version dilatée
        tailles = ndimage.sum(binaire, etiquettes, range(1, n + 1))
        binaire = binaire & (etiquettes == int(np.argmax(tailles)) + 1)

    ys, xs = np.nonzero(binaire)
    h, w = binaire.shape[:2]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    mx, my = int((x1 - x0) * marge), int((y1 - y0) * marge)
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


# À incrémenter dès que le recadrage change de comportement (correction EXIF, plus grande
# composante connexe, marge...). Le cache est indexé par chemin seulement : sans ce garde-fou,
# un changement de logique laisse des recadrages périmés que plus rien ne signale, et le
# catalogue se retrouve encodé autrement que les requêtes de l'API (mesuré : 0.22 de similarité
# entre les deux versions d'une même photo).
VERSION_RECADRAGE = 2


def recadrer_dossier(chemins: list[str], racine: Path, dossier_cache: Path, kind: str = "frames") -> list[str]:
    """Recadre chaque image sur la monture détectée et met le résultat en cache sur disque.
    Retourne un chemin par image d'entrée (l'original copié tel quel si la détection échoue,
    pour ne perdre aucune image de l'évaluation)."""
    from glasses_detector import GlassesSegmenter

    dossier_cache.mkdir(parents=True, exist_ok=True)
    jeton = dossier_cache / ".version_recadrage"
    version_cache = jeton.read_text().strip() if jeton.exists() else None
    if version_cache != str(VERSION_RECADRAGE):
        perimes = [p for p in dossier_cache.rglob("*") if p.is_file() and p != jeton]
        if perimes:
            print(f"[cache de recadrage obsolète -> {len(perimes)} images à recalculer]")
            for p in perimes:
                p.unlink()
        jeton.write_text(str(VERSION_RECADRAGE))
    segmenter = GlassesSegmenter(kind=kind, device=choisir_device())
    racine_abs = racine.resolve()

    sorties, echecs = [], 0
    for i, chemin in enumerate(chemins):
        rel = Path(chemin).resolve().relative_to(racine_abs)
        cible = dossier_cache / rel
        if not cible.exists():
            cible.parent.mkdir(parents=True, exist_ok=True)
            image = ouvrir_image(chemin)
            try:
                # On passe l'image déjà redressée, pas le chemin : sinon le détecteur relit le
                # fichier brut et travaille sur une monture couchée -> détection ratée.
                masque = np.array(segmenter.predict(image, format="mask"))
                boite = boite_depuis_masque(masque)
                if boite is None:
                    raise ValueError("masque vide")
                image.crop(boite).save(cible)
            except Exception:
                echecs += 1
                image.save(cible)  # repli : copie brute, pour ne pas bloquer la grille
        sorties.append(str(cible))
        if i % 200 == 0:
            print(f"  recadrage {i}/{len(chemins)}")
    if echecs:
        print(f"[glasses-detector] {echecs}/{len(chemins)} images non détectées -> copiées sans recadrage")
    return sorties


# ---------------------------------------------------------------- programme
def executer_grille(chemins_bruts, labels, backbones, conditions, cache_dir: Path,
                     racine: Path, batch: int, recalculer: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    chemins_recadres = None
    if "recadre" in conditions:
        try:
            chemins_recadres = recadrer_dossier(chemins_bruts, racine, cache_dir / "crops")
        except ImportError:
            print("[glasses-detector non installé -> pip install glasses-detector ; condition 'recadre' ignorée]")
            conditions = [c for c in conditions if c != "recadre"]

    resultats = []
    for nom_backbone in backbones:
        conf = BACKBONES[nom_backbone]
        for cond in conditions:
            chemins = chemins_bruts if cond == "brut" else chemins_recadres
            cle = f"{nom_backbone}_{cond}"
            cache = cache_dir / f"emb_{cle}.npz"

            if cache.exists() and not recalculer:
                d = np.load(cache, allow_pickle=True)  # compat avec d'anciens caches en dtype object
                emb, labels_cond = d["emb"], d["labels"]
                print(f"[{cle}] relu depuis le cache")
            else:
                print(f"[{cle}] calcul des embeddings ({len(chemins)} images)")
                emb, gardes = calculer(chemins, conf["famille"], conf["modele"], batch, conf.get("pooling"))
                labels_cond = np.asarray(labels)[gardes].astype(str)
                np.savez_compressed(cache, emb=emb, labels=labels_cond)

            res = evaluer_recall(emb, labels_cond)
            ligne = {"backbone": nom_backbone, "recadrage": cond}
            ligne.update({f"recall@{k}": round(res["recall"][k], 3) for k in KS})
            resultats.append(ligne)
            print(f"  -> recall@5 = {res['recall'][5]:.3f}\n")

    return pd.DataFrame(resultats).sort_values("recall@5", ascending=False).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/kaggle_glasses"))
    p.add_argument("--profondeur", type=int, default=2)
    p.add_argument("--max-classes", type=int, default=200)
    p.add_argument("--max-par-classe", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    p.add_argument("--sans-recadrage", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("data/grille"))
    p.add_argument("--recalculer", action="store_true")
    args = p.parse_args()

    df = lister_images(args.racine, args.profondeur)
    df = filtrer(df, args.max_classes, args.max_par_classe, args.seed)
    print(f"{len(df)} images, {df['label'].nunique()} classes retenues\n")

    conditions = ["brut"] if args.sans_recadrage else ["brut", "recadre"]
    tableau = executer_grille(
        df["chemin"].tolist(), df["label"].to_numpy(), args.backbones, conditions,
        args.cache_dir, args.racine, args.batch, args.recalculer,
    )

    print("=" * 60)
    print("Résumé de la grille (trié par recall@5 décroissant) :")
    print(tableau.to_string(index=False))
    sortie = args.cache_dir / "resume_grille.csv"
    tableau.to_csv(sortie, index=False)
    print(f"\nDétail : {sortie}")


if __name__ == "__main__":
    main()
