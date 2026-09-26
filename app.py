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
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from inventaire import Inventaire
from journal import Journal, chronometre
import metadonnees
from parse_etiquette import parse_etiquette
from recall_grid import BACKBONE_PROD, charger_backbone
from fiches import Fiches
from reseau import Reseau
from stock import Stock

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
    """Les k MONTURES les plus proches, par similarité cosinus (tout est normalisé -> produit
    scalaire).

    k montures, pas k photos. Le catalogue en compte trois à cinq par monture ; sans
    déduplication, une requête renvoyait « 18, 18, 18, 18, 18, 11 » -- cinq vignettes
    identiques, et une seule véritable alternative offerte à l'opticien. Chaque monture est
    représentée par sa meilleure photo, ce qui est aussi le score qui l'a classée.
    """
    if len(emb_catalogue) == 0:
        return []
    sims = emb_catalogue @ embedding_requete
    resultats: dict[str, float] = {}
    for i in np.argsort(-sims):
        label = str(labels_catalogue[i])
        if label not in resultats:
            resultats[label] = round(float(sims[i]), 4)
            if len(resultats) == k:
                break
    return [{"monture": m, "similarite": s} for m, s in resultats.items()]


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

    # La marque accompagne chaque résultat : elle permet d'enchaîner sur une recherche réseau
    # (module 3) sans que l'opticien ait à la retrouver lui-même.
    if modeles.marques_catalogue is not None:
        marques = {str(l): str(m).strip() for l, m in zip(modeles.labels_catalogue,
                                                           modeles.marques_catalogue)}
        for r in resultats:
            r["marque"] = marques.get(r["monture"], "")

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
_DONNEES = Path(os.environ.get("OPTI_STOCK_JOURNAL", "data/journal"))
JOURNAL = Journal(_DONNEES)
# Le réseau porte des messages entre entreprises : il va dans le même volume persistant.
RESEAU = Reseau(_DONNEES / "reseau.jsonl")

# --- Cloisonnement par boutique -------------------------------------------
# Stock, fiches et inventaires appartiennent à UNE boutique : un opticien qui s'inscrit doit
# trouver son propre rayon, pas celui du confrère qui a essayé l'application avant lui.
#
# Le cloisonnement passe par un dossier distinct plutôt que par un champ « boutique » filtré à
# la lecture. Un filtre oublié dans une requête laisse fuir les données du voisin sans que
# rien ne le signale ; un chemin séparé rend la fuite structurellement impossible, et les
# classes Stock, Fiches et Inventaire restent inchangées.
#
# Le journal des prédictions, lui, reste commun : il mesure le MODÈLE et non une boutique, et
# le fragmenter par boutique retarderait d'autant la seule mesure de fiabilité réelle.
_BOUTIQUES = _DONNEES / "boutiques"
# Les photos de fiches sont hors des dossiers de boutique : une balise <img> ne peut pas
# porter d'en-tête d'authentification. Elles sont donc adressées par leur seul identifiant,
# tiré au hasard. Les fiches elles-mêmes restent cloisonnées.
_PHOTOS_FICHES = _DONNEES / "photos_fiches"


def _dossier(boutique: str) -> Path:
    return _BOUTIQUES / boutique


def stock_de(boutique: str) -> Stock:
    return Stock(_dossier(boutique) / "stock.jsonl")


def fiches_de(boutique: str) -> Fiches:
    return Fiches(_dossier(boutique), photos=_PHOTOS_FICHES)


def inventaires_de(boutique: str) -> Inventaire:
    return Inventaire(_dossier(boutique) / "inventaires")


def _fiche_globale(monture: str) -> Path | None:
    """Photo d'une fiche, quelle que soit la boutique qui l'a saisie -- voir _PHOTOS_FICHES."""
    chemin = _PHOTOS_FICHES / f"{monture}.jpg"
    return chemin if chemin.is_file() else None


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
    # Monture saisie au comptoir : sa photo est rangée sous son identifiant, lui-même produit
    # par le serveur. Aucune valeur venue du client ne compose le chemin.
    if label.startswith("f_"):
        if (photo := _fiche_globale(label)) is None:
            raise HTTPException(status_code=404, detail=f"Aucune photo pour {label}")
        return FileResponse(photo, media_type="image/jpeg")

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


@app.post("/inventaire")
def route_inventaire_demarrer(request: Request, libelle: str = Query("", description="ex. 2026"),
                               x_boutique_jeton: str | None = Header(None)) -> dict:
    """Ouvre une session en figeant le stock théorique, pour qu'il puisse bouger pendant
    l'inventaire -- une vente, une réception -- sans fausser la comparaison finale."""
    b = _boutique(x_boutique_jeton)
    theorique = {m: e["quantite"] for m, e in stock_de(b).etat().items()}
    inventaires = inventaires_de(b)
    return inventaires.etat(inventaires.demarrer(theorique, libelle))


@app.get("/inventaire")
def route_inventaires(request: Request,
                       x_boutique_jeton: str | None = Header(None)) -> list[dict]:
    b = _boutique(x_boutique_jeton)
    identites = _identite(request, b)
    return [_nommer_ecarts(e, identites) for e in inventaires_de(b).sessions()]


def _nommer_ecarts(etat: dict, identites: dict[str, dict]) -> dict:
    """Remplace les identifiants techniques par les libellés dans les listes d'écarts. Une
    ligne « f_a1b2c3 manque » n'aide personne devant un rayon."""
    for cle in ("lignes", "manquantes", "en_trop"):
        etat[cle] = [{**l, "libelle": identites.get(l["monture"], {}).get("libelle",
                                                                          l["monture"])}
                     for l in etat.get(cle, [])]
    etat["hors_stock"] = [identites.get(m, {}).get("libelle", m)
                          for m in etat.get("hors_stock", [])]
    return etat


@app.get("/inventaire/{session}")
def route_inventaire_etat(request: Request, session: str,
                           x_boutique_jeton: str | None = Header(None)) -> dict:
    b = _boutique(x_boutique_jeton)
    try:
        return _nommer_ecarts(inventaires_de(b).etat(session), _identite(request, b))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Inventaire {session} inconnu")


@app.post("/inventaire/{session}/compter")
def route_inventaire_compter(request: Request, session: str,
                              monture: str = Query(..., description="Monture pointée"),
                              quantite: int = Query(1, ge=1, le=999),
                              x_boutique_jeton: str | None = Header(None)) -> dict:
    """Pointe une monture comme présente en rayon. L'identification par photo passe par
    /identifier ; ici l'opticien confirme, c'est lui qui fait foi."""
    b = _boutique(x_boutique_jeton)
    try:
        return _nommer_ecarts(inventaires_de(b).compter(session, monture, quantite),
                               _identite(request, b))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Inventaire {session} inconnu")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/inventaire/{session}/annuler")
def route_inventaire_annuler(request: Request, session: str, monture: str = Query(...),
                              quantite: int = Query(1, ge=1, le=999),
                              x_boutique_jeton: str | None = Header(None)) -> dict:
    b = _boutique(x_boutique_jeton)
    try:
        return _nommer_ecarts(inventaires_de(b).annuler_comptage(session, monture, quantite),
                               _identite(request, b))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Inventaire {session} inconnu")


@app.post("/inventaire/{session}/cloturer")
def route_inventaire_cloturer(
    request: Request,
    session: str,
    forcer: bool = Query(False, description="Clôturer même sans avoir rien compté, ce qui "
                                             "met tout le stock à zéro"),
    x_boutique_jeton: str | None = Header(None),
) -> dict:
    """Clôt et applique les comptages au stock : le rayon fait foi."""
    b = _boutique(x_boutique_jeton)
    try:
        return _nommer_ecarts(inventaires_de(b).cloturer(session, stock_de(b), forcer),
                               _identite(request, b))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Inventaire {session} inconnu")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------------------------------------------------------------------------
# Saisie d'une monture au comptoir (module 1)
#
# Le parcours qui ne dépend d'aucun modèle : photographier la monture, renseigner sa marque et
# sa référence -- à la main ou dictées par l'OCR de l'étiquette -- et l'entrer en stock. Il
# fonctionne donc sur n'importe quelle monture, y compris celles qu'aucun catalogue ne connaît.
# ---------------------------------------------------------------------------


@app.post("/monture")
async def route_creer_monture(
    photo: UploadFile | None = File(None, description="Photo de la monture"),
    marque: str = Query(..., min_length=1, max_length=80),
    reference: str = Query("", max_length=80),
    coloris: str = Query("", max_length=40),
    calibre: str = Query("", max_length=20),
    pont: str = Query("", max_length=20),
    branche: str = Query("", max_length=20),
    quantite: int = Query(1, ge=0, le=999, description="0 pour créer la fiche sans stock"),
    emplacement: str = Query("", max_length=80),
    x_boutique_jeton: str | None = Header(None),
) -> dict:
    """Crée la fiche d'une monture et l'entre en stock dans la foulée.

    Les deux gestes sont réunis parce qu'au comptoir ils n'en font qu'un : on ne saisit une
    monture que parce qu'on vient de la recevoir. `quantite=0` permet quand même de créer la
    fiche seule, par exemple pour préparer un réassort.
    """
    donnees = await photo.read() if photo is not None else None
    if donnees:
        _lire_image_uploadee(donnees)   # rejette tout de suite un fichier qui n'est pas une image

    b = _boutique(x_boutique_jeton)
    fiches = fiches_de(b)
    try:
        fiche = fiches.creer({"marque": marque, "reference": reference, "coloris": coloris,
                               "calibre": calibre, "pont": pont, "branche": branche}, donnees)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    etat = {"monture": fiche["monture"], "quantite": 0, "emplacement": ""}
    if quantite:
        etat = stock_de(b).entrer(fiche["monture"], quantite, emplacement, motif="reception")
    return {"fiche": fiche, "libelle": fiches.libelle(fiche["monture"]), "stock": etat}


@app.get("/monture")
def route_fiches(q: str = Query("", max_length=80, description="Recherche libre"),
                  x_boutique_jeton: str | None = Header(None)) -> list[dict]:
    """Fiches saisies, avec leur quantité en rayon."""
    b = _boutique(x_boutique_jeton)
    FICHES, etats = fiches_de(b), stock_de(b).etat()
    return [{**f, "libelle": FICHES.libelle(f["monture"]),
             "quantite": etats.get(f["monture"], {}).get("quantite", 0),
             "emplacement": etats.get(f["monture"], {}).get("emplacement", "")}
            for f in FICHES.chercher(q)]


@app.patch("/monture/{monture}")
def route_modifier_fiche(monture: str,
                          marque: str | None = Query(None, max_length=80),
                          reference: str | None = Query(None, max_length=80),
                          coloris: str | None = Query(None, max_length=40),
                          calibre: str | None = Query(None, max_length=20),
                          x_boutique_jeton: str | None = Header(None)) -> dict:
    champs = {c: v for c, v in (("marque", marque), ("reference", reference),
                                 ("coloris", coloris), ("calibre", calibre)) if v is not None}
    try:
        return fiches_de(_boutique(x_boutique_jeton)).modifier(monture, champs)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Fiche {monture} inconnue")


# ---------------------------------------------------------------------------
# Stock : ce qu'il y a en rayon, et où (module 1)
# ---------------------------------------------------------------------------


def _identite(request: Request, boutique: str) -> dict[str, dict]:
    """Ce qu'il faut savoir d'une monture pour l'afficher : marque, référence, libellé.

    Rassemble les deux origines -- le catalogue entraîné et les fiches saisies au comptoir --
    parce qu'en rayon personne ne reconnaît « f_a1b2c3 » ni « 18 ». Tout écran qui nomme une
    monture passe par ici : le stock comme l'inventaire, qui affichait encore des
    identifiants techniques dans sa liste d'écarts.
    """
    m = _modeles(request)
    marques = {}
    if m.marques_catalogue is not None:
        marques = {str(l): str(mq).strip()
                   for l, mq in zip(m.labels_catalogue, m.marques_catalogue)}
    fiches = fiches_de(boutique).toutes()
    references = _references_catalogue()

    identites: dict[str, dict] = {}
    for monture in set(marques) | set(fiches) | set(stock_de(boutique).etat()):
        fiche = fiches.get(monture, {})
        marque = fiche.get("marque") or marques.get(monture, "")
        reference = fiche.get("reference") or references.get(monture, "")
        # L'identifiant ne complète le libellé que pour une monture du catalogue sans
        # référence : c'est alors le seul repère qui reste.
        libelle = " ".join(x for x in (marque, reference or
                                        (monture if monture not in fiches else "")) if x)
        identites[monture] = {"marque": marque, "reference": reference,
                               "libelle": libelle or monture, "saisie": monture in fiches}
    return identites


@app.get("/stock")
def route_stock(request: Request, x_boutique_jeton: str | None = Header(None)) -> dict:
    """Le stock de CETTE boutique, nommé par ce qui est écrit sur les montures."""
    b = _boutique(x_boutique_jeton)
    stock, identites = stock_de(b), _identite(request, b)
    lignes = [{**e, **identites.get(monture, {"libelle": monture})}
              for monture, e in sorted(stock.etat().items())]
    return {"lignes": lignes, "bilan": stock.bilan()}


@app.get("/stock/{monture:path}")
def route_stock_monture(monture: str, x_boutique_jeton: str | None = Header(None)) -> dict:
    stock = stock_de(_boutique(x_boutique_jeton))
    return {**stock.etat_monture(monture), "mouvements": stock.mouvements(monture)}


@app.post("/stock/{monture:path}/entree")
def route_stock_entree(monture: str,
                        quantite: int = Query(1, ge=1, le=999),
                        emplacement: str = Query("", max_length=80),
                        motif: str = Query("reception"),
                        x_boutique_jeton: str | None = Header(None)) -> dict:
    try:
        return stock_de(_boutique(x_boutique_jeton)).entrer(monture, quantite, emplacement, motif)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stock/{monture:path}/sortie")
def route_stock_sortie(monture: str,
                        quantite: int = Query(1, ge=1, le=999),
                        motif: str = Query("vente"),
                        x_boutique_jeton: str | None = Header(None)) -> dict:
    """Refuse de descendre sous zéro : un stock négatif n'existe pas en rayon, et l'accepter
    rendrait tous les chiffres douteux. Le correctif est /ajuster."""
    try:
        return stock_de(_boutique(x_boutique_jeton)).sortir(monture, quantite, motif)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/stock/{monture:path}/ajuster")
def route_stock_ajuster(monture: str,
                         quantite: int = Query(..., ge=0, le=999),
                         motif: str = Query("correction"),
                         x_boutique_jeton: str | None = Header(None)) -> dict:
    try:
        return stock_de(_boutique(x_boutique_jeton)).ajuster(monture, quantite, motif)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/journal/bilan")
def route_bilan() -> dict:
    """recall@k mesuré sur les usages réels validés par l'opticien -- le seul chiffre qui ne
    souffre d'aucun des biais du jeu de photos initial."""
    return JOURNAL.bilan()


@app.get("/journal/seuil")
def route_seuil() -> dict:
    """Le seuil en service tient-il face à l'usage réel ?

    Lecture seule : recalibrer modifie ce que l'API affirme, et cela passe par
    `python apprendre.py --appliquer-seuil`, jamais par une requête web. La route sert à
    alerter -- un seuil calibré sur des photos d'une seule séance est probablement trop
    permissif en boutique, et c'est invisible sans cette comparaison.
    """
    from apprendre import seuil_reel
    return seuil_reel(JOURNAL, Path("data/tete.json"))


@app.post("/lire-etiquette")
async def route_lire_etiquette(
    request: Request,
    photo: UploadFile = File(..., description="Photo de l'étiquette (intérieur de branche)"),
) -> dict:
    image = _lire_image_uploadee(await photo.read())
    return lire_etiquette(image, _modeles(request))


# ---------------------------------------------------------------------------
# Saisie des marques
#
# 33 montures sur 115 n'ont pas de marque, et l'information n'est pas dans les photos : l'OCR
# n'y lit que les autocollants de verres, la gravure étant à l'intérieur de la branche. C'est
# donc une saisie humaine -- mais l'opticien reconnaît ses propres montures à l'œil, d'où une
# page qui montre la vignette plutôt qu'une liste d'identifiants.
# ---------------------------------------------------------------------------

METADONNEES = Path(os.environ.get("OPTI_STOCK_METADONNEES", "data/montures.csv"))


@app.get("/saisie", include_in_schema=False)
def route_saisie() -> FileResponse:
    page = Path(__file__).parent / "saisie.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="saisie.html introuvable")
    return FileResponse(page, media_type="text/html")


@app.get("/saisie/montures")
def route_saisie_montures() -> dict:
    """Ce qu'il reste à renseigner, plus l'état du stock pour situer l'effort."""
    return {"montures": metadonnees.a_completer(METADONNEES),
            "bilan": metadonnees.bilan(METADONNEES)}


@app.post("/saisie/marques")
def route_saisie_enregistrer(saisies: dict[str, dict]) -> dict:
    """Enregistre les marques saisies. `saisies` : {monture: {marque, reference, confirmee}}.

    `confirmee` dit qu'un humain a relu cette valeur : sans lui, une lecture OCR renvoyée telle
    quelle reste marquée « à vérifier » plutôt que de passer pour un fait établi.

    Écrit directement dans data/montures.csv. La modification ne change RIEN au modèle servi
    tant que `python mettre_a_jour.py` n'a pas été relancé : le catalogue embarque les marques
    au moment où il est construit. L'interface le rappelle après chaque enregistrement."""
    marques = {m: (v.get("marque") or "") for m, v in saisies.items()}
    references = {m: (v.get("reference") or "") for m, v in saisies.items()}
    confirmees = {m for m, v in saisies.items() if v.get("confirmee")}
    try:
        modifiees = metadonnees.ecrire_marques(METADONNEES, marques, references, confirmees)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"modifiees": modifiees, "bilan": metadonnees.bilan(METADONNEES)}


# ---------------------------------------------------------------------------
# Module 3 : réseau entre opticiens
#
# IDENTIFICATION DE DÉMONSTRATION. L'en-tête X-Boutique-Jeton suffit à agir au nom d'une
# boutique : pas de mot de passe, pas de vérification d'e-mail ni de SIRET, et le jeton
# circule en clair sans HTTPS en local. Cela suffit à montrer le parcours ; cela ne suffit
# pas à porter de vrais échanges entre entreprises. Voir l'en-tête de reseau.py.
# ---------------------------------------------------------------------------


def _boutique(jeton: str | None) -> str:
    """Identifie la boutique appelante, ou 401. Un seul endroit pour ce contrôle : c'est ici
    qu'il faudra brancher l'authentification externe."""
    if not jeton or (boutique := RESEAU.authentifier(jeton)) is None:
        raise HTTPException(status_code=401, detail="Jeton de boutique invalide ou absent")
    return boutique


def _catalogue_partageable(request: Request, boutique: str) -> list[dict]:
    """Les montures qu'on peut proposer au réseau : celles qui ont une marque ET qu'il reste en
    rayon.

    La marque, parce qu'aucun confrère ne saurait reconnaître une étiquette locale (« 50 »).
    Le stock, parce que proposer une monture vendue la semaine dernière fait déplacer quelqu'un
    pour rien -- c'est exactement ce que le réseau ne doit pas faire.
    """
    m = _modeles(request)
    references = _references_catalogue()
    marques_par_photo = (m.marques_catalogue if m.marques_catalogue is not None
                         else [""] * len(m.labels_catalogue))

    en_rayon = stock_de(boutique).en_stock()

    # Une monture par entrée, pas une par photo : le catalogue compte plusieurs vues de la même
    # monture, et la lister cinq fois rendrait le choix des montures à partager illisible.
    montures: dict[str, dict] = {}
    for label, marque in zip(m.labels_catalogue, marques_par_photo):
        label = str(label)
        if str(marque).strip() and label not in montures and label in en_rayon:
            montures[label] = {"libelle": label, "marque": str(marque).strip(),
                                "reference": references.get(label, ""),
                                "quantite": en_rayon[label]["quantite"]}

    # Les montures saisies au comptoir sont proposables aussi : ce sont souvent les seules que
    # la boutique possède vraiment, et elles portent déjà marque et référence.
    for identifiant, fiche in fiches_de(boutique).toutes().items():
        if fiche.get("marque") and identifiant in en_rayon:
            montures[identifiant] = {"libelle": identifiant, "marque": fiche["marque"],
                                      "reference": fiche.get("reference", ""),
                                      "quantite": en_rayon[identifiant]["quantite"]}
    return sorted(montures.values(), key=lambda x: (x["marque"], x["libelle"]))


def _references_catalogue() -> dict[str, str]:
    """Références lues dans data/montures.csv. Le catalogue .npz ne les porte pas : elles ne
    servent qu'au réseau, où elles sont la seule clé commune entre deux boutiques."""
    fichier = Path("data/montures.csv")
    if not fichier.is_file():
        return {}
    import csv
    with open(fichier, newline="", encoding="utf-8") as f:
        return {(l.get("monture") or "").strip(): (l.get("reference") or "").strip()
                for l in csv.DictReader(f) if (l.get("reference") or "").strip()}


@app.post("/reseau/inscription")
def route_inscription(
    nom: str = Query(..., min_length=2, max_length=120, description="Nom de la boutique"),
    ville: str = Query("", max_length=80),
    email: str = Query("", max_length=160, description="Contact, non diffusé aux confrères"),
) -> dict:
    """Inscrit une boutique et lui remet son jeton. Le jeton n'est affiché qu'ici : il vaut
    mot de passe, et n'est pas récupérable ensuite (rien ne le stocke en clair côté client)."""
    return RESEAU.inscrire(nom.strip(), ville.strip(), email.strip())


@app.get("/reseau/moi")
def route_moi(x_boutique_jeton: str | None = Header(None)) -> dict:
    """La boutique que désigne ce jeton, ou 401. Sert à se reconnecter depuis un autre
    téléphone : on ne croit pas le code sur parole, c'est le serveur qui tranche."""
    return RESEAU.boutiques()[_boutique(x_boutique_jeton)]


@app.get("/reseau/boutiques")
def route_boutiques() -> list[dict]:
    """Annuaire : nom et ville uniquement. Ni jeton ni e-mail."""
    return sorted(RESEAU.boutiques().values(), key=lambda b: b.get("nom", ""))


@app.get("/reseau/mon-stock")
def route_mon_stock(request: Request,
                    x_boutique_jeton: str | None = Header(None)) -> dict:
    """Ce que cette boutique *peut* proposer au réseau, et ce qu'elle propose déjà."""
    boutique = _boutique(x_boutique_jeton)
    partageables = _catalogue_partageable(request, boutique)
    deja = {m.get("libelle") for m in RESEAU.partage(boutique)}
    return {"boutique": boutique,
            "montures": [{**m, "partagee": m["libelle"] in deja} for m in partageables]}


@app.post("/reseau/partage")
def route_partage(request: Request,
                  montures: list[str] = Query([], description="Étiquettes locales à proposer"),
                  x_boutique_jeton: str | None = Header(None)) -> dict:
    """Déclare ce que la boutique propose au réseau. Remplace le partage précédent : envoyer
    une liste vide retire tout. Le partage est un choix explicite, monture par monture -- le
    reste du stock n'est jamais exposé."""
    boutique = _boutique(x_boutique_jeton)
    voulues = set(montures)
    a_partager = [m for m in _catalogue_partageable(request, boutique) if m["libelle"] in voulues]

    inconnues = voulues - {m["libelle"] for m in a_partager}
    if inconnues:
        raise HTTPException(status_code=400,
                            detail=f"Montures absentes du stock ou sans marque : "
                                   f"{', '.join(sorted(inconnues)[:5])}")

    return {"boutique": boutique, "partagees": RESEAU.partager(boutique, a_partager)}


@app.get("/reseau/chercher")
def route_chercher(marque: str = Query("", description="Marque recherchée"),
                   reference: str = Query("", description="Référence, si elle est connue"),
                   x_boutique_jeton: str | None = Header(None)) -> list[dict]:
    """Qui, dans le réseau, propose cette monture. Réservé aux boutiques inscrites : cet
    annuaire dit qui détient quel stock, il n'a pas à être public."""
    boutique = _boutique(x_boutique_jeton)
    if not marque.strip() and not reference.strip():
        raise HTTPException(status_code=400, detail="Préciser au moins une marque ou une référence")

    # La disponibilité se vérifie ICI, pas au moment du partage. Le partage est un choix
    # durable figé dans le journal ; le stock, lui, bouge. Filtrer seulement à l'inscription
    # laisserait proposer une monture vendue depuis -- le confrère se déplace pour rien.
    #
    # Et c'est le rayon de CHAQUE boutique proposante qu'on interroge, pas celui du chercheur :
    # chaque boutique a le sien depuis qu'elles sont cloisonnées.
    rayons: dict[str, dict] = {}
    trouves = []
    for t in RESEAU.chercher(marque=marque, reference=reference, sauf_boutique=boutique):
        proprietaire = t["boutique"]
        if proprietaire not in rayons:
            rayons[proprietaire] = stock_de(proprietaire).en_stock()
        if t.get("libelle") in rayons[proprietaire]:
            trouves.append({**t, "quantite": rayons[proprietaire][t["libelle"]]["quantite"]})
    return trouves


@app.post("/reseau/demande")
def route_demande(boutique_sollicitee: str = Query(..., description="À qui s'adresser"),
                  monture: str = Query("", description="ex. « OCTIKA OS866 »"),
                  texte: str = Query(..., min_length=1, max_length=2000),
                  x_boutique_jeton: str | None = Header(None)) -> dict:
    """Ouvre la conversation et envoie le premier message, en une fois : c'est le geste réel de
    l'opticien qui vient de trouver une monture chez un confrère."""
    demandeuse = _boutique(x_boutique_jeton)
    if boutique_sollicitee not in RESEAU.boutiques():
        raise HTTPException(status_code=404, detail="Boutique inconnue")
    try:
        conversation = RESEAU.ouvrir_conversation(demandeuse, boutique_sollicitee, monture)
        RESEAU.ecrire(conversation, demandeuse, texte)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"conversation": conversation}


@app.get("/reseau/conversations")
def route_conversations(x_boutique_jeton: str | None = Header(None)) -> list[dict]:
    return RESEAU.conversations(_boutique(x_boutique_jeton))


@app.get("/reseau/conversation/{conversation}")
def route_messages(conversation: str,
                   x_boutique_jeton: str | None = Header(None)) -> dict:
    boutique = _boutique(x_boutique_jeton)
    try:
        messages = RESEAU.messages(conversation, boutique)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation inconnue")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    noms = RESEAU.boutiques()
    return {"conversation": conversation, "moi": boutique,
            "messages": [{**m, "nom": noms.get(m["auteur"], {}).get("nom", "?")}
                         for m in messages]}


@app.post("/reseau/conversation/{conversation}/message")
def route_repondre(conversation: str,
                   texte: str = Query(..., min_length=1, max_length=2000),
                   x_boutique_jeton: str | None = Header(None)) -> dict:
    boutique = _boutique(x_boutique_jeton)
    try:
        return RESEAU.ecrire(conversation, boutique, texte)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation inconnue")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reseau/bloquer")
def route_bloquer(boutique_bloquee: str = Query(...),
                  x_boutique_jeton: str | None = Header(None)) -> dict:
    """Une boutique bloquée ne peut plus écrire et ne voit plus le stock partagé du bloqueur.
    Réversible, et non réciproque."""
    boutique = _boutique(x_boutique_jeton)
    try:
        RESEAU.bloquer(boutique, boutique_bloquee)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"bloquee": boutique_bloquee}


@app.post("/reseau/signaler")
def route_signaler(conversation: str = Query(...),
                   motif: str = Query(..., min_length=1, max_length=500),
                   x_boutique_jeton: str | None = Header(None)) -> dict:
    """Consigne un signalement. Il est conservé, pas traité : la modération humaine reste à
    organiser avant toute ouverture du réseau."""
    boutique = _boutique(x_boutique_jeton)
    try:
        RESEAU.messages(conversation, boutique)   # participant ? sinon rien à signaler
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation inconnue")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    RESEAU.signaler(conversation, boutique, motif)
    return {"signale": conversation}
