"""API opti-stock : identification de monture par similarité + lecture d'étiquette par OCR.

Regroupe les modules 1 (entrée en stock par photo) et 4 (reconnaissance de monture) du projet.
Les modules 2 (inventaire) et 3 (réseau entre opticiens) ne sont pas dans cette API.

Installation :
    pip install fastapi "uvicorn[standard]" python-multipart torch transformers pillow numpy \
                pandas easyocr glasses-detector

Fichiers attendus :
    data/catalogue.npz   -- catalogue de référence (construire_catalogue.py)
    data/tete.pt         -- tête de projection (entrainer_tete.py) ; facultative, l'API
                            retombe sur les embeddings bruts du backbone si elle est absente

Lancement (développement) :
    uvicorn app:app --reload --port 8000

Puis, par exemple :
    curl -X POST http://localhost:8000/identifier -F "photo=@monture.jpg"
    curl -X POST http://localhost:8000/lire-etiquette -F "photo=@etiquette.jpg"

Documentation interactive une fois lancé : http://localhost:8000/docs
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from journal import Journal, chronometre
from parse_etiquette import parse_etiquette
from recall_grid import BACKBONE_PROD, boite_depuis_masque, charger_backbone

# Seuil de similarité en dessous duquel l'API ne présente pas de réponse. Mesuré sur 115
# montures, avec des photos de requête jamais vues à l'entraînement de la tête :
#     sans seuil   100 % de réponses, précision@5 = 0.35
#     seuil 0.80    20 % de réponses, précision@5 = 0.79
#     seuil 0.85    12 % de réponses, précision@5 = 0.86
# Le seuil n'a de sens qu'avec la tête de projection : sur les embeddings bruts du backbone les
# scores ne sont pas calibrés (précision@1 plafonne à 0.58 même en ne répondant que 10 % du
# temps). À remesurer quand le stock change d'échelle -- la précision dépend du nombre de
# montures en concurrence.
SEUIL_CONFIANCE = 0.85


# ---------------------------------------------------------------- modèles chargés
@dataclass
class Modeles:
    """Regroupe tout ce dont les endpoints ont besoin. Les champs sont des fonctions
    simples (image -> résultat), pour pouvoir les remplacer facilement par des versions
    factices dans les tests, sans dépendre des vrais modèles."""
    recadrer: Callable[[Image.Image], Image.Image]
    embedder: Callable[[list[Image.Image]], np.ndarray]  # images -> embeddings backbone normalisés
    tete: Callable[[np.ndarray], np.ndarray]              # embeddings bruts -> embeddings projetés normalisés
    ocr: Callable[[Image.Image], str]
    marques_connues: list[str]
    emb_catalogue: np.ndarray   # (M, D_projete), déjà passés par la tête, normalisés
    labels_catalogue: np.ndarray  # (M,)
    marques_catalogue: np.ndarray | None = None  # (M,), "" quand la marque est inconnue


def charger_modeles(
    catalogue_path: Path = Path("data/catalogue.npz"),
    tete_path: Path = Path("data/tete.pt"),
    dim_sortie: int = 128,
) -> Modeles:
    """Charge le backbone, le détecteur de monture, l'OCR, la tête de projection et le catalogue
    de référence. Coûteux -> appelé une seule fois, au démarrage de l'API (voir `lifespan` plus
    bas), jamais par requête."""
    import torch
    from easyocr import Reader
    from glasses_detector import GlassesSegmenter

    from finetune_triplet import TeteProjection

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    embedder = charger_backbone(BACKBONE_PROD, device)
    segmenter = GlassesSegmenter(kind="frames", device=device)
    lecteur_ocr = Reader(["fr", "en"], gpu=(device != "cpu"))

    def recadrer(image: Image.Image) -> Image.Image:
        masque = np.array(segmenter.predict(image, format="mask"))
        boite = boite_depuis_masque(masque)
        return image.crop(boite) if boite is not None else image

    def ocr(image: Image.Image) -> str:
        return " ".join(lecteur_ocr.readtext(np.array(image), detail=0, paragraph=False))

    d = np.load(catalogue_path, allow_pickle=True)
    # Un catalogue encodé avec un autre backbone que les requêtes donne des résultats absurdes
    # sans lever la moindre erreur : on refuse explicitement plutôt que de servir du bruit.
    backbone_catalogue = str(d["backbone"]) if "backbone" in d else "(non précisé)"
    if backbone_catalogue != BACKBONE_PROD:
        raise RuntimeError(
            f"{catalogue_path} a été construit avec le backbone '{backbone_catalogue}' alors que "
            f"l'API encode avec '{BACKBONE_PROD}'. Relancer construire_catalogue.py."
        )

    if tete_path.exists():
        tete_module = TeteProjection(d["emb"].shape[1], dim_sortie=dim_sortie).to(device)
        tete_module.load_state_dict(torch.load(tete_path, map_location=device))
        tete_module.eval()

        def tete(emb_bruts: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                y = tete_module(torch.tensor(emb_bruts, dtype=torch.float32, device=device))
            return y.cpu().numpy().astype("float32")
    else:
        def tete(emb_bruts: np.ndarray) -> np.ndarray:
            return emb_bruts

    emb_catalogue = tete(d["emb"])
    labels_catalogue = d["labels"]
    marques_catalogue = d["marques"] if "marques" in d else None

    # Les marques viennent des métadonnées quand elles existent. À défaut on retombe sur les
    # identifiants de monture, qui ne sont pas des marques : /lire-etiquette ne sait alors pas
    # reconnaître de marque, faute d'une liste à comparer.
    if marques_catalogue is not None:
        marques_connues = sorted({str(m).strip() for m in marques_catalogue if str(m).strip()})
    else:
        marques_connues = []
    if not marques_connues:
        marques_connues = sorted(set(str(l).split("/")[0] for l in labels_catalogue))

    return Modeles(
        recadrer=recadrer, embedder=embedder, tete=tete, ocr=ocr,
        marques_connues=marques_connues,
        emb_catalogue=emb_catalogue, labels_catalogue=labels_catalogue,
        marques_catalogue=marques_catalogue,
    )


# ---------------------------------------------------------------- logique métier (testable sans les vrais modèles)
def recherche_topk(embedding_requete: np.ndarray, emb_catalogue: np.ndarray,
                    labels_catalogue: np.ndarray, k: int = 5) -> list[dict]:
    """Plus proches voisins par similarité cosinus (tout est déjà normalisé -> produit scalaire)."""
    if len(emb_catalogue) == 0:
        return []
    k = min(k, len(emb_catalogue))
    sims = emb_catalogue @ embedding_requete
    ordre = np.argsort(-sims)[:k]
    return [{"monture": str(labels_catalogue[i]), "similarite": round(float(sims[i]), 4)} for i in ordre]


def _masque_marque(modeles: Modeles, marque: str) -> np.ndarray | None:
    """Indices du catalogue appartenant à `marque`. None si le filtre n'est pas applicable
    (aucune métadonnée, ou marque absente du stock) -- on cherche alors dans tout le catalogue
    plutôt que de ne rien renvoyer."""
    if not marque or modeles.marques_catalogue is None:
        return None
    voulue = marque.strip().casefold()
    garde = np.array([str(m).strip().casefold() == voulue for m in modeles.marques_catalogue])
    return garde if garde.any() else None


def identifier_monture(image: Image.Image, modeles: Modeles, k: int = 5,
                        seuil: float = SEUIL_CONFIANCE, marque: str = "") -> dict:
    """`fiable` dit si la réponse mérite d'être présentée comme une identification. En dessous du
    seuil les candidats sont quand même renvoyés -- un humain peut reconnaître la bonne monture
    dans une liste que le modèle n'assume pas -- mais l'interface ne doit pas les présenter comme
    une réponse : sans ce garde-fou, la première proposition est fausse 3 fois sur 4, avec un
    score d'apparence crédible, ce qui peut faire commander la mauvaise référence."""
    recadree = modeles.recadrer(image)
    emb_brut = modeles.embedder([recadree])
    emb_projete = modeles.tete(emb_brut)[0]

    # Restreindre les candidats à une marque aide beaucoup : mesuré sur ce stock, le recall@5
    # passe de 0.35 avec 115 montures en concurrence à 0.49 avec 40.
    garde = _masque_marque(modeles, marque)
    emb_cat = modeles.emb_catalogue if garde is None else modeles.emb_catalogue[garde]
    labels_cat = modeles.labels_catalogue if garde is None else modeles.labels_catalogue[garde]

    resultats = recherche_topk(emb_projete, emb_cat, labels_cat, k)
    return {
        "resultats": resultats,
        "fiable": bool(resultats) and resultats[0]["similarite"] >= seuil,
        "seuil": seuil,
        "marque_filtree": marque if garde is not None else "",
        "candidats": int(len(labels_cat)),
    }


def lire_etiquette(image: Image.Image, modeles: Modeles) -> dict:
    texte = modeles.ocr(image)
    e = parse_etiquette(texte, modeles.marques_connues)
    return {
        "texte_brut": texte,
        "marque": e.marque,
        "reference": e.reference,
        "coloris_code": e.coloris_code,
        "calibre": e.calibre,
        "pont": e.pont,
        "branche": e.branche,
    }


def _lire_image_uploadee(donnees: bytes) -> Image.Image:
    try:
        # exif_transpose : les photos de téléphone arrivent non pivotées, avec un tag EXIF.
        return ImageOps.exif_transpose(Image.open(io.BytesIO(donnees))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier image invalide : {e}")


# ---------------------------------------------------------------- application FastAPI
# CHARGEUR_MODELES est une variable de module (pas un appel direct dans lifespan) pour
# pouvoir la remplacer par une version factice dans les tests, sans toucher au reste.
CHARGEUR_MODELES: Callable[[], Modeles] = charger_modeles
JOURNAL = Journal(Path("data/journal"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modeles = CHARGEUR_MODELES()
    yield


app = FastAPI(
    title="Opti-Stock API",
    description="Identification de monture par similarité (module 4) et lecture d'étiquette par OCR (module 1)",
    lifespan=lifespan,
)

# CORS ouvert : nécessaire pour que demo.html (ouvert en local, sans serveur) appelle l'API.
# À restreindre à une liste de domaines précis avant toute mise en production réelle.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _modeles(request: Request) -> Modeles:
    m = getattr(request.app.state, "modeles", None)
    if m is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés (l'API démarre encore)")
    return m


@app.get("/sante")
def sante() -> dict:
    return {"statut": "ok"}


@app.post("/identifier")
async def route_identifier(
    request: Request,
    photo: UploadFile = File(..., description="Photo de la monture à identifier"),
    k: int = Query(5, ge=1, le=50, description="Nombre de résultats à renvoyer"),
    seuil: float = Query(SEUIL_CONFIANCE, ge=0.0, le=1.0,
                          description="En dessous, la réponse est marquée non fiable"),
    marque: str = Query("", description="Restreint la recherche à cette marque, si connue"),
) -> dict:
    donnees = await photo.read()
    image = _lire_image_uploadee(donnees)
    horloge = chronometre()
    resultat = identifier_monture(image, _modeles(request), k, seuil, marque)
    resultat["journal_id"] = JOURNAL.enregistrer(
        type_prediction="similarite", resultat=resultat, version=BACKBONE_PROD,
        photo=donnees, latence_ms=horloge(),
    )
    return resultat


@app.post("/journal/{journal_id}/choix")
def route_choix(journal_id: str, monture: str = Query(..., description="La bonne monture")) -> dict:
    """Déclare la bonne réponse pour une prédiction. C'est ce qui transforme un usage en cas de
    test étiqueté : photo réelle, conditions réelles, vérité terrain humaine."""
    if not JOURNAL.noter_choix(journal_id, monture):
        raise HTTPException(status_code=404, detail=f"Prédiction {journal_id} inconnue")
    return {"journal_id": journal_id, "monture_choisie": monture}


@app.get("/journal/bilan")
def route_bilan() -> dict:
    """recall@k mesuré sur les usages réels validés par l'opticien -- le seul chiffre qui ne
    souffre d'aucun des biais du jeu de photos initial."""
    return JOURNAL.bilan()


@app.post("/lire-etiquette")
async def route_lire_etiquette(
    request: Request,
    photo: UploadFile = File(..., description="Photo de l'étiquette (intérieur de branche)"),
) -> dict:
    image = _lire_image_uploadee(await photo.read())
    return lire_etiquette(image, _modeles(request))
