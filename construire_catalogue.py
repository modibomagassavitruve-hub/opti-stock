"""Construit le catalogue de référence de l'API à partir des VRAIES montures en stock.

Produit data/catalogue.npz, le fichier que /identifier interroge. Le backbone utilisé
(BACKBONE_PROD) est inscrit dans le fichier, et charger_modeles refuse un catalogue bâti avec
un autre : un catalogue encodé différemment des requêtes donnerait des résultats absurdes sans
rien signaler.

Arborescence attendue (une photo peut suffire, mais prends-en plusieurs : face, 3/4, profil,
étiquette) :
    data/mes_montures/<identifiant_monture>/photo1.jpg
    data/mes_montures/<identifiant_monture>/photo2.jpg
    ...
L'identifiant peut être ce que tu veux, du moment qu'il est unique par monture et cohérent
entre ses photos (c'est lui qui sera renvoyé par /identifier comme résultat).

Usage :
    python construire_catalogue.py
    python construire_catalogue.py --racine data/mes_montures --recadrage

Ajout d'une nouvelle monture au stock : dépose son dossier de photos sous --racine, puis
relance ce script (pas de mise à jour incrémentale pour l'instant, volontairement simple tant
que le stock tient en quelques centaines de montures -- tout est recalculé à chaque lancement)
et redémarre l'API pour qu'elle recharge le nouveau catalogue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from recall_clip import choisir_device, lister_images, ouvrir_image
from recall_grid import BACKBONE_PROD, charger_backbone, recadrer_dossier


def lire_metadonnees(chemin: Path | None, montures: set[str]) -> dict[str, str]:
    """Lit un CSV `monture,marque` et renvoie la marque de chaque monture (vide si inconnue).

    La marque ne peut pas être extraite des photos de façon fiable : mesuré sur 115 montures,
    l'OCR n'en identifie que 30 % de façon confirmée. Elle vient donc d'ici -- du système de
    stock, ou saisie à l'entrée en stock."""
    marques = {m: "" for m in montures}
    if chemin is None or not chemin.exists():
        return marques

    import csv

    inconnues = []
    with open(chemin, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            monture = (ligne.get("monture") or "").strip()
            if monture in marques:
                marques[monture] = (ligne.get("marque") or "").strip()
            elif monture:
                inconnues.append(monture)
    if inconnues:
        print(f"[métadonnées] {len(inconnues)} monture(s) du CSV absente(s) du stock, ignorée(s) : "
              f"{', '.join(inconnues[:5])}{'...' if len(inconnues) > 5 else ''}")
    renseignees = sum(1 for v in marques.values() if v)
    print(f"[métadonnées] marque connue pour {renseignees}/{len(marques)} montures")
    return marques


RATIO_MONTURE = 2.2   # une monture vue de face occupe un cadre ~2 fois plus large que haut
COTE_VIGNETTE = 420


def _forme_utilisable(chemin: Path) -> tuple:
    """Clé de tri : d'abord la forme la plus proche d'une monture de face, puis la plus grande.
    Sert à choisir, parmi les photos d'une monture, celle où l'opticien la reconnaîtra."""
    try:
        with Image.open(chemin) as im:
            l, h = im.size
    except Exception:
        return (float("inf"), 0)
    return (abs(max(l / h, h / l) - RATIO_MONTURE), -(l * h))


def generer_vignettes(chemins: np.ndarray, labels: np.ndarray, dossier: Path) -> int:
    """Écrit une petite vignette par monture, pour l'interface.

    Sans elles, l'API servirait les photos d'origine : ~2 Mo chacune, soit une dizaine de Mo par
    recherche, et elles sont exclues de l'image Docker. Une vignette de 420 px pèse ~40 Ko et
    voyage avec le catalogue."""
    dossier.mkdir(parents=True, exist_ok=True)
    for ancienne in dossier.glob("*.jpg"):
        ancienne.unlink()

    echecs = 0
    for label in sorted(set(labels.tolist())):
        candidats = [Path(c) for c, l in zip(chemins, labels) if l == label]
        candidats = [c for c in candidats if c.is_file()]
        if not candidats:
            continue
        try:
            im = ouvrir_image(min(candidats, key=_forme_utilisable))
            im.thumbnail((COTE_VIGNETTE, COTE_VIGNETTE))
            im.save(dossier / f"{label}.jpg", quality=82)
        except Exception:
            # Une vignette est cosmétique : son échec ne doit pas empêcher la construction du
            # catalogue, qui est ce dont l'API a réellement besoin. L'interface retombera sur
            # la photo d'origine pour cette monture.
            echecs += 1
    if echecs:
        print(f"[vignettes] {echecs} monture(s) sans vignette, l'interface servira l'original")
    return len(list(dossier.glob("*.jpg")))


def construire(
    racine: Path,
    sortie: Path,
    recadrage: bool = False,
    batch: int = 16,
    dossier_crops: Path | None = None,
    metadonnees: Path | None = None,
) -> dict | None:
    """Construit le catalogue et l'écrit dans `sortie` (mêmes clés que attend app.py :
    emb, labels). Retourne un résumé, ou None si aucune photo n'a été trouvée.

    `recadrage` est désactivé par défaut : mesuré sur 115 montures et 3 découpages, recadrer
    sur la monture détectée FAIT CHUTER le recall@5 de 0.80 à 0.33. Le détecteur rate 29 % des
    photos -- bandes très allongées, fragments minuscules -- et ces recadrages détruisent
    l'information. L'image entière, elle, contient toujours la monture. Vérifié par le contrôle
    à décor équivalent : le signal d'identité est meilleur sans recadrage (0.74 contre 0.61),
    donc le gain ne vient pas du fond partagé entre photos d'une même rafale."""
    df = lister_images(racine, profondeur=1)
    if df.empty:
        print(f"Aucune photo trouvée sous {racine}")
        print(f"Arborescence attendue : {racine}/<id_monture>/photo.jpg")
        return None
    print(f"{len(df)} photos, {df['label'].nunique()} montures retenues")

    chemins = df["chemin"].tolist()
    if recadrage:
        cache = dossier_crops or (racine.parent / f"{racine.name}_crops")
        chemins = recadrer_dossier(chemins, racine, cache)

    print(f"Calcul des embeddings {BACKBONE_PROD} ({len(chemins)} photos)...")
    encoder = charger_backbone(BACKBONE_PROD, choisir_device())
    vecteurs, gardes = [], []
    for debut in range(0, len(chemins), batch):
        images, idx = [], []
        for j, chemin in enumerate(chemins[debut: debut + batch]):
            try:
                images.append(ouvrir_image(chemin))
                idx.append(debut + j)
            except Exception as e:
                print(f"[ignorée] {chemin}: {e}")
        if images:
            vecteurs.append(encoder(images))
            gardes.extend(idx)
        if (debut // batch) % 20 == 0:
            print(f"  {min(debut + batch, len(chemins))}/{len(chemins)}")

    emb = np.concatenate(vecteurs)
    gardes = np.array(gardes)
    labels = df["label"].to_numpy()[gardes].astype(str)
    chemins_gardes = np.array(chemins)[gardes].astype(str)

    par_monture = lire_metadonnees(metadonnees, set(labels.tolist()))
    marques = np.array([par_monture.get(l, "") for l in labels])

    n_vignettes = generer_vignettes(chemins_gardes, labels, sortie.parent / "vignettes")
    print(f"{n_vignettes} vignettes écrites dans {sortie.parent / 'vignettes'}")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sortie, emb=emb, labels=labels, chemins=chemins_gardes,
                        marques=marques, backbone=np.array(BACKBONE_PROD))
    return {"n_photos": len(emb), "n_montures": len(set(labels)), "sortie": sortie}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/mes_montures"))
    p.add_argument("--recadrage", action="store_true",
                    help="recadre sur la monture détectée ; dégrade fortement le recall "
                         "(0.80 -> 0.33 mesuré), conservé pour comparaison")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sortie", type=Path, default=Path("data/catalogue.npz"))
    p.add_argument("--crops", type=Path, default=None,
                    help="cache des recadrages (défaut : <racine>_crops, à côté des photos)")
    p.add_argument("--metadonnees", type=Path, default=Path("data/montures.csv"),
                    help="CSV monture,marque -- permet de filtrer la recherche par marque")
    args = p.parse_args()

    resume = construire(args.racine, args.sortie, args.recadrage, args.batch, args.crops,
                        args.metadonnees)
    if resume is None:
        return

    print(f"\n{resume['n_photos']} photos indexées, {resume['n_montures']} montures -> {resume['sortie']}")
    print("Redémarre l'API (uvicorn) pour qu'elle recharge ce catalogue.")


if __name__ == "__main__":
    main()
