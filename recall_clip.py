"""Premier recall@k : recherche d'images de lunettes avec CLIP (zero-shot, sans entraînement).

Principe : chaque image sert tour à tour de requête ; on cherche ses voisins les plus
proches parmi toutes les AUTRES images. Succès si l'un des k premiers résultats
appartient au même produit (= même dossier). C'est exactement ton cas d'usage :
"je photographie une monture, retrouve-t-on la même dans le stock ?"

Installation (dans l'environnement opti-stock) :
    pip install torch transformers pillow numpy pandas

Données : dataset Kaggle "Glasses dataset" (egorovlvan/glasses-dataset), à dézipper dans
data/kaggle_glasses. Structure supposée : un dossier par produit contenant ses photos.

Usage :
    python recall_clip.py --racine data/kaggle_glasses --explorer     # 1) inspecter la structure
    python recall_clip.py --racine data/kaggle_glasses                # 2) évaluer (200 classes)
    python recall_clip.py --racine data/kaggle_glasses --max-classes 1000 --max-par-classe 30
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
KS = (1, 5, 10)


def ouvrir_image(chemin):
    """Ouvre une image en appliquant son orientation EXIF. Indispensable : les photos de
    téléphone sont stockées non pivotées, avec un tag EXIF qui indique la rotation à appliquer.
    PIL ne l'applique pas tout seul -> sans ça, montures couchées sur le côté, ce qui casse à la
    fois la détection de monture et les embeddings (modèles entraînés sur des photos droites)."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(chemin)).convert("RGB")


# ---------------------------------------------------------------- données
def _ignore(chemin_relatif: Path) -> bool:
    return any(p.startswith(".") or p == "__MACOSX" for p in chemin_relatif.parts)


def lister_images(racine: Path, profondeur: int = 1) -> pd.DataFrame:
    """Liste les images ; le label = les `profondeur` premiers dossiers sous la racine."""
    lignes = []
    for chemin in sorted(racine.rglob("*")):
        if not chemin.is_file() or chemin.suffix.lower() not in EXTENSIONS:
            continue
        rel = chemin.relative_to(racine)
        if _ignore(rel) or len(rel.parts) <= profondeur:
            continue  # image ignorée ou posée sans dossier produit
        lignes.append({"chemin": str(chemin), "label": "/".join(rel.parts[:profondeur])})
    return pd.DataFrame(lignes, columns=["chemin", "label"])


def filtrer(df: pd.DataFrame, max_classes: int, max_par_classe: int, seed: int = 0) -> pd.DataFrame:
    """Garde les classes d'au moins 2 images, en échantillonnant pour un test rapide."""
    tailles = df.groupby("label").size()
    valides = tailles[tailles >= 2].index.to_numpy()
    rng = np.random.default_rng(seed)
    if len(valides) > max_classes:
        valides = rng.choice(valides, size=max_classes, replace=False)
    df = df[df["label"].isin(valides)].sample(frac=1, random_state=seed)  # mélange reproductible
    return (
        df.groupby("label").head(max_par_classe)
        .sort_values(["label", "chemin"])
        .reset_index(drop=True)
    )


def explorer(racine: Path, df: pd.DataFrame) -> None:
    extensions = Counter(p.suffix.lower() for p in racine.rglob("*") if p.is_file())
    print("Types de fichiers :", dict(extensions.most_common(8)))
    print(f"{len(df)} images, {df['label'].nunique()} classes (profondeur choisie)")
    if len(df):
        print("Images par classe :")
        print(df.groupby("label").size().describe().round(1).to_string())
        print("\nExemples de chemins :")
        for c in df["chemin"].head(5):
            print("  ", Path(c).relative_to(racine))
        print("\nSi les labels ne ressemblent pas à un produit, relance avec --profondeur 2.")


# ---------------------------------------------------------------- embeddings
def _tenseur(sortie):
    import torch

    if isinstance(sortie, torch.Tensor):
        return sortie
    for nom in ("image_embeds", "pooler_output"):
        valeur = getattr(sortie, nom, None)
        if valeur is not None:
            return valeur
    raise TypeError(f"Sortie CLIP inattendue : {type(sortie)}")


def choisir_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def calculer_embeddings(chemins: list[str], nom_modele: str, batch: int = 32):
    """Retourne (embeddings normalisés float32, indices des images lues correctement)."""
    import torch
    from transformers import CLIPImageProcessor, CLIPModel

    device = choisir_device()
    print(f"Modèle {nom_modele} sur {device}")
    modele = CLIPModel.from_pretrained(nom_modele).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(nom_modele)

    vecteurs, gardes = [], []
    with torch.no_grad():
        for debut in range(0, len(chemins), batch):
            images, idx = [], []
            for j, chemin in enumerate(chemins[debut: debut + batch]):
                try:
                    images.append(ouvrir_image(chemin))
                    idx.append(debut + j)
                except Exception as e:  # image corrompue
                    print(f"[ignorée] {chemin}: {e}")
            if not images:
                continue
            entree = proc(images=images, return_tensors="pt").to(device)
            emb = _tenseur(modele.get_image_features(**entree))
            emb = torch.nn.functional.normalize(emb, dim=-1)
            vecteurs.append(emb.cpu().numpy().astype("float32"))
            gardes.extend(idx)
            if (debut // batch) % 20 == 0:
                print(f"  {min(debut + batch, len(chemins))}/{len(chemins)}")
    return np.concatenate(vecteurs), np.array(gardes)


# ---------------------------------------------------------------- évaluation
def evaluer_recall(emb: np.ndarray, labels, ks=KS, chunk: int = 1024) -> dict:
    """Recall@k en leave-one-out + niveau du hasard, par requête."""
    labels = np.asarray(labels)
    n = len(labels)
    kmax = min(max(ks), n - 1)
    _, inverse = np.unique(labels, return_inverse=True)
    taille = np.bincount(inverse)

    top_idx = np.zeros((n, kmax), dtype=np.int64)
    top_sim = np.zeros((n, kmax), dtype=np.float32)
    for debut in range(0, n, chunk):
        fin = min(n, debut + chunk)
        sim = emb[debut:fin] @ emb.T
        sim[np.arange(fin - debut), np.arange(debut, fin)] = -np.inf  # exclut la requête
        idx = np.argpartition(-sim, kmax - 1, axis=1)[:, :kmax]
        s = np.take_along_axis(sim, idx, axis=1)
        ordre = np.argsort(-s, axis=1)
        top_idx[debut:fin] = np.take_along_axis(idx, ordre, axis=1)
        top_sim[debut:fin] = np.take_along_axis(s, ordre, axis=1)

    memes = inverse[top_idx] == inverse[:, None]
    hits = {k: memes[:, : min(k, kmax)].any(axis=1) for k in ks}

    # Hasard : probabilité qu'au moins un des k tirages sans remise soit de la même classe.
    m = (taille[inverse] - 1).astype(float)
    n1 = float(n - 1)
    hasard = {}
    for k in ks:
        p_aucun = np.ones(n)
        for i in range(min(k, n - 1)):
            p_aucun *= np.clip(n1 - m - i, 0, None) / (n1 - i)
        hasard[k] = float((1 - p_aucun).mean())

    return {
        "recall": {k: float(h.mean()) for k, h in hits.items()},
        "hasard": hasard,
        "hits": hits,
        "top_idx": top_idx,
        "top_sim": top_sim,
    }


# ---------------------------------------------------------------- programme
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/kaggle_glasses"))
    p.add_argument("--profondeur", type=int, default=1)
    p.add_argument("--explorer", action="store_true", help="affiche la structure et s'arrête")
    p.add_argument("--max-classes", type=int, default=200)
    p.add_argument("--max-par-classe", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modele", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cache", type=Path, default=Path("data/embeddings_clip.npz"))
    p.add_argument("--recalculer", action="store_true")
    p.add_argument("--sortie", type=Path, default=Path("data/recall_clip.csv"))
    args = p.parse_args()

    df = lister_images(args.racine, args.profondeur)
    if args.explorer or df.empty:
        explorer(args.racine, df)
        return

    df = filtrer(df, args.max_classes, args.max_par_classe, args.seed)
    chemins = df["chemin"].tolist()
    print(f"{len(df)} images, {df['label'].nunique()} classes retenues")

    emb = None
    if args.cache.exists() and not args.recalculer:
        cache = np.load(args.cache, allow_pickle=False)
        if np.array_equal(cache["chemins"], np.array(chemins)) and str(cache["modele"]) == args.modele:
            emb = cache["emb"]
            print("Embeddings relus depuis le cache")
    if emb is None:
        emb, gardes = calculer_embeddings(chemins, args.modele, args.batch)
        df = df.iloc[gardes].reset_index(drop=True)
        chemins = df["chemin"].tolist()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, emb=emb, chemins=np.array(chemins), modele=args.modele)

    res = evaluer_recall(emb, df["label"].to_numpy())
    print(f"\nRésultats sur {len(df)} requêtes (CLIP zero-shot, {args.modele}) :")
    for k in KS:
        print(f"  Recall@{k:<2} : {res['recall'][k]:.3f}   (hasard : {res['hasard'][k]:.3f})")

    top = res["top_idx"]
    detail = pd.DataFrame(
        {
            "chemin_requete": chemins,
            "label": df["label"],
            **{f"hit@{k}": res["hits"][k] for k in KS},
            "top1_chemin": [chemins[i] for i in top[:, 0]],
            "top1_label": df["label"].to_numpy()[top[:, 0]],
            "top1_similarite": res["top_sim"][:, 0].round(4),
        }
    )
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.sortie, index=False)
    print(f"\nDétail par requête : {args.sortie}  (filtre hit@5 == False pour analyser les échecs)")


if __name__ == "__main__":
    main()
