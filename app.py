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
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from journal import Journal, chronometre
from parse_etiquette import parse_etiquette
from recall_grid import BACKBONE_PROD, charger_backbone

# Seuil de repli, utilisé seulement si la tête n'apporte pas le sien. entrainer_tete.py calibre
# le seuil à chaque entraînement et l'écrit à côté du modèle (data/tete.json), car sa bonne
# valeur dépend de la distribution des scores -- laquelle change avec le backbone, le
# prétraitement et l'augmentation. Écrit en dur, il a dû être corrigé à la main quatre fois et
# restait faux entre-temps, sans que rien ne le signale.
SEUIL_PAR_DEFAUT = 0.80

# Vignettes produites par construire_catalogue.py, servies à l'interface.
DOSSIER_VIGNETTES = Path("data/vignettes")


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
    seuil_confiance: float = SEUIL_PAR_DEFAUT   # calibré à l'entraînement, lu dans tete.json
    marques_catalogue: np.ndarray | None = None  # (M,), "" quand la marque est inconnue
    chemins_catalogue: np.ndarray | None = None  # (M,), pour servir une vignette au client


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

    from finetune_triplet import TeteProjection

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    embedder = charger_backbone(BACKBONE_PROD, device)
    lecteur_ocr = Reader(["fr", "en"], gpu=(device != "cpu"))

    def recadrer(image: Image.Image) -> Image.Image:
        # Le recadrage sur la monture détectée est DÉSACTIVÉ : mesuré sur 115 montures et
        # 3 découpages, il fait chuter le recall@5 de 0.80 à 0.33. Le détecteur rate 29 % des
        # photos (bandes très allongées, fragments minuscules) et ces recadrages détruisent
        # l'information, alors que l'image entière contient toujours la monture.
        # Le catalogue est construit de la même façon (construire_catalogue.py) : requêtes et
        # catalogue DOIVENT subir le même traitement, sinon les embeddings ne se comparent pas.
        return image

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

    # Seuil calibré lors de l'entraînement, à côté du modèle. Sa bonne valeur dépend de la
    # distribution des scores de CETTE tête : la lire ici évite qu'une constante du code se
    # désynchronise du modèle servi.
    seuil = SEUIL_PAR_DEFAUT
    fichier_seuil = tete_path.with_suffix(".json")
    if fichier_seuil.exists():
        import json

        seuil = float(json.loads(fichier_seuil.read_text(encoding="utf-8"))["seuil_confiance"])

    # Une tête plus ancienne que le catalogue a été entraînée sans les montures ajoutées depuis,
    # et son seuil est calibré sur un stock qui n'existe plus. Rien n'échoue -- les dimensions
    # restent compatibles -- donc seul un avertissement peut l'attraper.
    if tete_path.exists() and tete_path.stat().st_mtime < catalogue_path.stat().st_mtime:
        print(f"[ATTENTION] {tete_path} est plus ancienne que {catalogue_path}. La tête ignore "
              f"les montures ajoutées depuis et son seuil de confiance est périmé. "
              f"Relancer : python entrainer_tete.py")

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
        seuil_confiance=seuil,
        marques_catalogue=marques_catalogue,
        chemins_catalogue=d["chemins"] if "chemins" in d else None,
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
                        seuil: float | None = None, marque: str = "") -> dict:
    """`fiable` dit si la réponse mérite d'être présentée comme une identification. En dessous du
    seuil les candidats sont quand même renvoyés -- un humain peut reconnaître la bonne monture
    dans une liste que le modèle n'assume pas -- mais l'interface ne doit pas les présenter comme
    une réponse : sans ce garde-fou, la première proposition est fausse 3 fois sur 4, avec un
    score d'apparence crédible, ce qui peut faire commander la mauvaise référence."""
    seuil = modeles.seuil_confiance if seuil is None else seuil
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


RATIO_MONTURE = 2.2  # une monture vue de face, bien recadrée, est ~2 fois plus large que haute


def _photo_la_plus_lisible(chemins: list[Path]) -> Path:
    """Choisit la photo qui montre le mieux la monture, pour que l'opticien la reconnaisse.

    Le détecteur produit parfois des recadrages inexploitables -- bandes très allongées ou
    fragments minuscules (mesuré : 28 % des recadrages ont un rapport de côtés supérieur à 3).
    On préfère donc la photo dont la forme se rapproche d'une monture vue de face, en écartant
    les vignettes trop petites pour être lues."""
    def defaut(chemin: Path) -> tuple:
        try:
            with Image.open(chemin) as im:
                l, h = im.size
        except Exception:
            return (float("inf"), 0)
        ratio = max(l / h, h / l)
        return (abs(ratio - RATIO_MONTURE), -(l * h) if l * h >= 50_000 else 0)

    return min(chemins, key=defaut)


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

# Le journal accumule les identifications validées par l'opticien : c'est la seule mesure de
# fiabilité prise en conditions réelles, et il se construit sur des semaines d'usage. En
# conteneur, l'écrire dans l'image le ferait disparaître à chaque redéploiement -- d'où le
# chemin configurable, à faire pointer vers un volume monté.
JOURNAL = Journal(Path(os.environ.get("OPTI_STOCK_JOURNAL", "data/journal")))


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


@app.get("/", include_in_schema=False)
def route_interface() -> FileResponse:
    """Sert l'interface depuis l'API : une seule adresse à ouvrir pour l'opticien, et pas de
    requête inter-origine à autoriser."""
    page = Path(__file__).parent / "demo.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="demo.html introuvable")
    return FileResponse(page, media_type="text/html")


@app.post("/identifier")
async def route_identifier(
    request: Request,
    photo: UploadFile = File(..., description="Photo de la monture à identifier"),
    k: int = Query(5, ge=1, le=50, description="Nombre de résultats à renvoyer"),
    seuil: float | None = Query(None, ge=0.0, le=1.0,
                                description="En dessous, la réponse est marquée non fiable. "
                                            "Par défaut : le seuil calibré à l'entraînement"),
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


@app.get("/marques")
def route_marques(request: Request) -> dict:
    """Marques présentes au catalogue, pour alimenter le filtre côté interface."""
    m = _modeles(request)
    if m.marques_catalogue is None:
        return {"marques": []}
    return {"marques": sorted({str(x).strip() for x in m.marques_catalogue if str(x).strip()})}


@app.get("/monture/{label:path}/photo")
def route_photo_monture(request: Request, label: str) -> FileResponse:
    """Vignette d'une monture du catalogue, pour que l'opticien reconnaisse visuellement.

    Le chemin n'est jamais construit à partir de l'entrée : `label` doit figurer parmi les
    étiquettes du catalogue, et on sert alors un chemin établi côté serveur. Une valeur
    fantaisiste, y compris une tentative de remontée de dossier, ne correspond à aucune
    étiquette et reçoit un 404."""
    m = _modeles(request)
    if label not in {str(l) for l in m.labels_catalogue}:
        raise HTTPException(status_code=404, detail=f"Monture {label} inconnue")

    # Vignette produite par construire_catalogue.py : ~40 Ko contre ~2 Mo pour l'originale,
    # et elle accompagne le catalogue là où les photos sources ne sont pas déployées.
    vignette = DOSSIER_VIGNETTES / f"{label}.jpg"
    if vignette.is_file():
        return FileResponse(vignette, media_type="image/jpeg")

    if m.chemins_catalogue is not None:
        sources = [Path(str(c)) for c, l in zip(m.chemins_catalogue, m.labels_catalogue)
                   if str(l) == label]
        sources = [c for c in sources if c.is_file()]
        if sources:
            return FileResponse(_photo_la_plus_lisible(sources), media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Aucune image pour la monture {label}")


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
