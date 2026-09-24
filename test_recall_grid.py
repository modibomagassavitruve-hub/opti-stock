from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PIL import Image

import recall_grid as rg


# ---------------------------------------------------------------- boite_depuis_masque
def test_boite_masque_vide_renvoie_none():
    assert rg.boite_depuis_masque(np.zeros((10, 10))) is None


def test_boite_masque_avec_marge():
    m = np.zeros((100, 100))
    m[40:60, 30:70] = 1  # bloc y:[40,59] x:[30,69]
    x0, y0, x1, y1 = rg.boite_depuis_masque(m, marge=0.1)
    # indices inclusifs : x0=30, x1=69 (largeur 39) ; y0=40, y1=59 (hauteur 19)
    # marge x = int(39*0.1) = 3 ; marge y = int(19*0.1) = 1
    assert (x0, y0, x1, y1) == (27, 39, 72, 60)


def test_boite_masque_marge_bornee_par_image():
    m = np.zeros((10, 10))
    m[0:10, 0:10] = 1  # occupe toute l'image : la marge ne doit pas sortir du cadre
    x0, y0, x1, y1 = rg.boite_depuis_masque(m, marge=0.2)
    assert (x0, y0, x1, y1) == (0, 0, 10, 10)


def test_cache_de_recadrage_invalide_par_changement_de_version(tmp_path, monkeypatch):
    """Régression : le cache est indexé par chemin seulement. Sans jeton de version, un
    changement de logique de recadrage laisse des images périmées, et le catalogue se retrouve
    encodé autrement que les requêtes de l'API — sans que rien ne le signale."""
    racine = tmp_path / "photos"
    (racine / "m1").mkdir(parents=True)
    source = racine / "m1" / "a.jpg"
    Image.new("RGB", (40, 40), "white").save(source)
    cache = tmp_path / "crops"

    appels = []

    class FauxSegmenter:
        def __init__(self, **kwargs):
            pass

        def predict(self, image, format):
            appels.append(1)
            m = np.zeros((40, 40))
            m[10:20, 10:20] = 1
            return m

    monkeypatch.setattr("glasses_detector.GlassesSegmenter", FauxSegmenter)

    rg.recadrer_dossier([str(source)], racine, cache)
    assert len(appels) == 1
    rg.recadrer_dossier([str(source)], racine, cache)
    assert len(appels) == 1, "le cache doit être réutilisé à version identique"

    monkeypatch.setattr(rg, "VERSION_RECADRAGE", rg.VERSION_RECADRAGE + 1)
    rg.recadrer_dossier([str(source)], racine, cache)
    assert len(appels) == 2, "un changement de version doit forcer le recalcul"


def test_boite_masque_ignore_les_montures_voisines():
    """Photo de présentoir : la monture visée (grand bloc) et une voisine (petit bloc à l'autre
    bout du cadre). La boîte ne doit couvrir que la monture visée, pas toute l'étagère."""
    m = np.zeros((200, 200))
    m[20:60, 20:80] = 1      # monture visée
    m[180:190, 180:195] = 1  # monture voisine, nettement plus petite
    assert rg.boite_depuis_masque(m, marge=0.0) == (20, 20, 79, 59)


def test_boite_masque_recolle_les_fragments_dune_meme_monture():
    """Monture métal fine : le masque ressort en morceaux disjoints mais proches. La dilatation
    doit les recoller, sinon on recadre sur un seul fragment."""
    m = np.zeros((200, 200))
    m[100:110, 40:60] = 1
    m[100:110, 68:88] = 1  # même monture, trou de 8 px
    x0, _, x1, _ = rg.boite_depuis_masque(m, marge=0.0)
    assert (x0, x1) == (40, 87)


# ---------------------------------------------------------------- executer_grille
def _fausse_calculer(labels_par_chemin):
    """Fabrique un faux `calculer` : embeddings alignés sur le vrai label -> recall parfait."""
    compteur = {"appels": 0}

    def calculer(chemins, famille, nom_modele, batch, pooling=None):
        compteur["appels"] += 1
        classes = sorted(set(labels_par_chemin[c] for c in chemins))
        base = np.eye(len(classes), dtype="float32")
        emb = np.array([base[classes.index(labels_par_chemin[c])] for c in chemins])
        return emb, np.arange(len(chemins))

    return calculer, compteur


def test_grille_brut_seul_et_resume_correct(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(8)]
    labels = ["a", "a", "b", "b", "c", "c", "d", "d"]
    par_chemin = dict(zip(chemins, labels))
    faux_calculer, compteur = _fausse_calculer(par_chemin)
    monkeypatch.setattr(rg, "calculer", faux_calculer)

    df = rg.executer_grille(
        chemins, labels, backbones=["clip", "dinov2"], conditions=["brut"],
        cache_dir=tmp_path, racine=tmp_path, batch=4, recalculer=False,
    )
    assert set(df["backbone"]) == {"clip", "dinov2"}
    assert (df["recall@5"] == 1.0).all()  # embeddings parfaitement séparés par classe
    assert compteur["appels"] == 2  # un appel par backbone
    assert (tmp_path / "emb_clip_brut.npz").exists()
    assert (tmp_path / "emb_dinov2_brut.npz").exists()


def test_grille_relit_le_cache_sans_recalculer(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(6)]
    labels = ["a", "a", "b", "b", "c", "c"]
    par_chemin = dict(zip(chemins, labels))
    faux_calculer, compteur = _fausse_calculer(par_chemin)
    monkeypatch.setattr(rg, "calculer", faux_calculer)

    rg.executer_grille(chemins, labels, ["clip"], ["brut"], tmp_path, tmp_path, 4, False)
    assert compteur["appels"] == 1
    rg.executer_grille(chemins, labels, ["clip"], ["brut"], tmp_path, tmp_path, 4, False)
    assert compteur["appels"] == 1  # deuxième appel : tout relu depuis le cache


def test_grille_recalculer_force_le_calcul(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(6)]
    labels = ["a", "a", "b", "b", "c", "c"]
    par_chemin = dict(zip(chemins, labels))
    faux_calculer, compteur = _fausse_calculer(par_chemin)
    monkeypatch.setattr(rg, "calculer", faux_calculer)

    rg.executer_grille(chemins, labels, ["clip"], ["brut"], tmp_path, tmp_path, 4, False)
    rg.executer_grille(chemins, labels, ["clip"], ["brut"], tmp_path, tmp_path, 4, recalculer=True)
    assert compteur["appels"] == 2


def test_grille_recadrage_absent_retombe_sur_brut(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(4)]
    labels = ["a", "a", "b", "b"]
    par_chemin = dict(zip(chemins, labels))
    faux_calculer, _ = _fausse_calculer(par_chemin)
    monkeypatch.setattr(rg, "calculer", faux_calculer)

    def leve_import_error(*a, **k):
        raise ImportError("glasses_detector non installé")

    monkeypatch.setattr(rg, "recadrer_dossier", leve_import_error)
    df = rg.executer_grille(
        chemins, labels, ["clip"], ["brut", "recadre"], tmp_path, tmp_path, 4, False,
    )
    # glasses-detector absent -> seule la condition 'brut' doit apparaître, pas de crash
    assert set(df["recadrage"]) == {"brut"}


def test_grille_resultat_trie_par_recall5_decroissant(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(6)]
    labels = ["a", "a", "b", "b", "c", "c"]

    def calculer_variable(chemins, famille, nom_modele, batch, pooling=None):
        n = len(chemins)
        if famille == "clip":  # bien séparé -> bon recall
            classes = sorted(set(labels))
            base = np.eye(len(classes), dtype="float32")
            emb = np.array([base[classes.index(labels[i % len(labels)])] for i in range(n)])
        else:  # aléatoire -> mauvais recall
            rng = np.random.default_rng(0)
            emb = rng.normal(size=(n, 8)).astype("float32")
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb, np.arange(n)

    monkeypatch.setattr(rg, "calculer", calculer_variable)
    df = rg.executer_grille(chemins, labels, ["clip", "dinov2"], ["brut"], tmp_path, tmp_path, 4, False)
    assert df.iloc[0]["backbone"] == "clip"  # le meilleur recall@5 est en tête
    assert df["recall@5"].is_monotonic_decreasing


def test_dinov2_mean_et_cls_sont_deux_backbones_distincts_dans_le_cache(tmp_path, monkeypatch):
    chemins = [f"img{i}.jpg" for i in range(4)]
    labels = ["a", "a", "b", "b"]
    par_chemin = dict(zip(chemins, labels))
    appels = []

    def faux_calculer(chemins, famille, nom_modele, batch, pooling=None):
        appels.append(pooling)
        classes = sorted(set(labels))
        base = np.eye(len(classes), dtype="float32")
        emb = np.array([base[classes.index(par_chemin[c])] for c in chemins])
        return emb, np.arange(len(chemins))

    monkeypatch.setattr(rg, "calculer", faux_calculer)
    df = rg.executer_grille(
        chemins, labels, ["dinov2", "dinov2_mean"], ["brut"], tmp_path, tmp_path, 4, False,
    )
    assert set(df["backbone"]) == {"dinov2", "dinov2_mean"}
    assert sorted(appels) == ["cls", "mean"]  # chaque variante appelée avec son propre pooling
    assert (tmp_path / "emb_dinov2_brut.npz").exists()
    assert (tmp_path / "emb_dinov2_mean_brut.npz").exists()
