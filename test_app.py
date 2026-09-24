import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
from app import Modeles, identifier_monture, lire_etiquette, recherche_topk


# ---------------------------------------------------------------- recherche_topk
def test_recherche_topk_ordre_par_similarite_decroissante():
    emb_cat = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype="float32")
    labels = np.array(["a", "b", "c"])
    requete = np.array([1.0, 0.0], dtype="float32")
    res = recherche_topk(requete, emb_cat, labels, k=2)
    assert [r["monture"] for r in res] == ["a", "b"]
    assert res[0]["similarite"] > res[1]["similarite"]


def test_recherche_topk_borne_k_a_la_taille_du_catalogue():
    emb_cat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    labels = np.array(["a", "b"])
    res = recherche_topk(np.array([1.0, 0.0], dtype="float32"), emb_cat, labels, k=50)
    assert len(res) == 2  # pas d'erreur même si k > taille du catalogue


def test_tete_perimee_signalee_au_demarrage(tmp_path, monkeypatch, capsys):
    """Catalogue reconstruit sans réentraîner la tête : rien n'échoue (les dimensions restent
    compatibles) mais la tête ignore les montures ajoutées et son seuil est périmé."""
    import os
    import time

    tete = tmp_path / "tete.pt"
    tete.write_bytes(b"")
    time.sleep(0.01)
    catalogue = tmp_path / "catalogue.npz"
    np.savez_compressed(catalogue, emb=np.eye(3, dtype="float32"),
                        labels=np.array(["a", "b", "c"]),
                        backbone=np.array(app_module.BACKBONE_PROD))
    os.utime(tete, (0, 0))  # tête nettement plus ancienne que le catalogue

    monkeypatch.setattr(app_module, "charger_backbone", lambda *a, **k: (lambda images: None))
    monkeypatch.setattr("easyocr.Reader", lambda *a, **k: None)
    monkeypatch.setattr("torch.load", lambda *a, **k: {})
    monkeypatch.setattr("finetune_triplet.TeteProjection.load_state_dict", lambda self, e: None)

    app_module.charger_modeles(catalogue_path=catalogue, tete_path=tete)
    assert "ATTENTION" in capsys.readouterr().out


def test_seuil_vient_du_modele_pas_du_code(tmp_path, monkeypatch):
    """Le seuil est calibré à l'entraînement et rangé à côté de la tête : une constante du code
    se désynchroniserait du modèle servi, ce qui est déjà arrivé quatre fois."""
    import json

    catalogue = tmp_path / "catalogue.npz"
    np.savez_compressed(catalogue, emb=np.eye(3, dtype="float32"),
                        labels=np.array(["a", "b", "c"]),
                        backbone=np.array(app_module.BACKBONE_PROD))
    tete = tmp_path / "tete.pt"
    tete.with_suffix(".json").write_text(json.dumps({"seuil_confiance": 0.123}), encoding="utf-8")

    monkeypatch.setattr(app_module, "charger_backbone", lambda *a, **k: (lambda images: None))
    monkeypatch.setattr("easyocr.Reader", lambda *a, **k: None)
    m = app_module.charger_modeles(catalogue_path=catalogue, tete_path=tete)
    assert m.seuil_confiance == 0.123


def test_seuil_par_defaut_si_le_modele_nen_fournit_pas():
    modeles = _faux_modeles()
    assert modeles.seuil_confiance == app_module.SEUIL_PAR_DEFAUT


def test_charger_modeles_refuse_un_catalogue_dun_autre_backbone(tmp_path):
    """Un catalogue encodé avec un autre backbone que les requêtes renverrait du bruit sans
    lever d'erreur : le démarrage doit échouer franchement plutôt que servir ça."""
    catalogue = tmp_path / "catalogue.npz"
    np.savez_compressed(catalogue, emb=np.eye(3, dtype="float32"),
                        labels=np.array(["a", "b", "c"]), backbone=np.array("un_autre_backbone"))
    with pytest.raises(RuntimeError, match="un_autre_backbone"):
        app_module.charger_modeles(catalogue_path=catalogue, tete_path=tmp_path / "absente.pt")


def test_recherche_topk_catalogue_vide():
    res = recherche_topk(np.array([1.0, 0.0], dtype="float32"),
                          np.empty((0, 2), dtype="float32"), np.array([]), k=5)
    assert res == []


# ---------------------------------------------------------------- Modeles factices
def _faux_modeles(catalogue_labels=("talla/bogart2", "visionario/mikel03", "talla/gravita9015")):
    emb_cat = np.eye(len(catalogue_labels), dtype="float32")

    def recadrer(image):
        return image  # pas de vrai recadrage dans les tests

    def embedder(images):
        # embedding déterministe et distinct par "couleur moyenne" -> suffisant pour tester le pipeline
        return np.array([[1.0, 0.0, 0.0][:2] + [0.0] * (len(catalogue_labels) - 2) for _ in images], dtype="float32")[:, :len(catalogue_labels)]

    def tete(emb_bruts):
        return emb_bruts  # identité : simplifie le test du branchement, pas la vraie tête

    def ocr(image):
        return "RAY-BAN RB3025 001/51 58 14 135"

    return Modeles(
        recadrer=recadrer, embedder=embedder, tete=tete, ocr=ocr,
        marques_connues=["Ray-Ban", "Talla", "Visionario"],
        emb_catalogue=emb_cat, labels_catalogue=np.array(catalogue_labels),
    )


# ---------------------------------------------------------------- identifier_monture / lire_etiquette (logique)
def _modeles_avec_marques():
    m = _faux_modeles(catalogue_labels=("m1", "m2", "m3"))
    m.marques_catalogue = np.array(["Osmose", "Maritza", "Osmose"])
    return m


def test_filtre_marque_restreint_les_candidats():
    m = _modeles_avec_marques()
    res = identifier_monture(Image.new("RGB", (5, 5)), m, k=5, seuil=0.0, marque="Osmose")
    assert res["candidats"] == 2
    assert res["marque_filtree"] == "Osmose"
    assert {r["monture"] for r in res["resultats"]} == {"m1", "m3"}


def test_filtre_marque_insensible_a_la_casse_et_aux_espaces():
    m = _modeles_avec_marques()
    res = identifier_monture(Image.new("RGB", (5, 5)), m, k=5, seuil=0.0, marque="  osmose ")
    assert res["candidats"] == 2


def test_marque_inconnue_cherche_dans_tout_le_catalogue():
    """Mieux vaut chercher partout que ne rien renvoyer : une marque absente du stock vient
    sans doute d'une faute de frappe ou d'une métadonnée manquante."""
    m = _modeles_avec_marques()
    res = identifier_monture(Image.new("RGB", (5, 5)), m, k=5, seuil=0.0, marque="Inexistante")
    assert res["candidats"] == 3
    assert res["marque_filtree"] == ""


def test_sans_metadonnees_le_filtre_est_ignore():
    m = _faux_modeles()  # marques_catalogue vaut None
    res = identifier_monture(Image.new("RGB", (5, 5)), m, k=5, seuil=0.0, marque="Osmose")
    assert res["candidats"] == 3
    assert res["marque_filtree"] == ""


def test_identifier_marque_non_fiable_sous_le_seuil():
    """Sans ce drapeau, la première proposition est fausse 3 fois sur 4 tout en affichant un
    score crédible — de quoi faire commander la mauvaise référence."""
    modeles = _faux_modeles()
    image = Image.new("RGB", (10, 10))
    # requête à mi-chemin entre deux entrées du catalogue -> similarité ~0.707
    modeles.embedder = lambda images: np.array([[0.707, 0.707, 0.0]], dtype="float32")

    haut = identifier_monture(image, modeles, k=1, seuil=0.5)
    assert haut["fiable"] is True
    assert haut["seuil"] == 0.5

    bas = identifier_monture(image, modeles, k=1, seuil=0.85)
    assert bas["fiable"] is False
    assert bas["resultats"], "les candidats restent renvoyés : un humain peut reconnaître la bonne"


def test_identifier_catalogue_vide_nest_jamais_fiable():
    modeles = _faux_modeles()
    modeles.emb_catalogue = np.empty((0, 3), dtype="float32")
    modeles.labels_catalogue = np.array([])
    res = identifier_monture(Image.new("RGB", (10, 10)), modeles, k=5, seuil=0.0)
    assert res["resultats"] == []
    assert res["fiable"] is False


def test_identifier_monture_renvoie_k_resultats():
    modeles = _faux_modeles()
    image = Image.new("RGB", (10, 10))
    res = identifier_monture(image, modeles, k=2)
    assert len(res["resultats"]) == 2
    assert set(res["resultats"][0]) == {"monture", "similarite"}


def test_identifier_monture_appelle_recadrage_avant_embedding(monkeypatch):
    ordre_appels = []
    modeles = _faux_modeles()

    def recadrer_trace(image):
        ordre_appels.append("recadrer")
        return image

    def embedder_trace(images):
        ordre_appels.append("embedder")
        return np.zeros((1, 3), dtype="float32")

    modeles.recadrer = recadrer_trace
    modeles.embedder = embedder_trace
    identifier_monture(Image.new("RGB", (5, 5)), modeles, k=1)
    assert ordre_appels == ["recadrer", "embedder"]  # le recadrage doit précéder l'embedding


def test_lire_etiquette_extrait_bien_les_champs():
    modeles = _faux_modeles()
    res = lire_etiquette(Image.new("RGB", (5, 5)), modeles)
    assert res["marque"] == "Ray-Ban"
    assert res["reference"] == "RB3025"
    assert res["coloris_code"] == "001/51"
    assert (res["calibre"], res["pont"], res["branche"]) == ("58", "14", "135")
    assert "texte_brut" in res


# ---------------------------------------------------------------- routes HTTP (bout en bout, sans vrais modèles)
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "CHARGEUR_MODELES", _faux_modeles)
    with TestClient(app_module.app) as c:  # déclenche lifespan -> CHARGEUR_MODELES() factice, pas de téléchargement
        yield c


def _fichier_image_valide() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color=(120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_route_marques_alimente_le_filtre(client, monkeypatch):
    client.app.state.modeles.marques_catalogue = np.array(["Osmose", "", "Maritza", "Osmose"])
    assert client.get("/marques").json() == {"marques": ["Maritza", "Osmose"]}


def test_route_marques_sans_metadonnees(client):
    client.app.state.modeles.marques_catalogue = None
    assert client.get("/marques").json() == {"marques": []}


def test_photo_monture_sert_la_vignette(client, tmp_path):
    image = tmp_path / "m.jpg"
    Image.new("RGB", (8, 8)).save(image)
    m = client.app.state.modeles
    m.chemins_catalogue = np.array([str(image)] * len(m.labels_catalogue))
    r = client.get(f"/monture/{m.labels_catalogue[0]}/photo")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_photo_monture_refuse_une_etiquette_inconnue(client, tmp_path):
    """Le chemin n'est jamais construit depuis l'entrée : une valeur fantaisiste, y compris une
    tentative de remontée de dossier, ne correspond à aucune étiquette du catalogue."""
    m = client.app.state.modeles
    m.chemins_catalogue = np.array(["/tmp/x.jpg"] * len(m.labels_catalogue))
    assert client.get("/monture/inconnue/photo").status_code == 404
    assert client.get("/monture/..%2F..%2Fetc%2Fpasswd/photo").status_code == 404


def test_route_sante(client):
    assert client.get("/sante").json() == {"statut": "ok"}


def test_route_identifier_ok(client):
    r = client.post("/identifier", files={"photo": ("monture.jpg", _fichier_image_valide(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "resultats" in body
    assert len(body["resultats"]) == 3  # k=5 demandé, mais catalogue factice de 3 montures -> plafonné


def test_route_identifier_respecte_le_parametre_k(client):
    r = client.post("/identifier?k=1", files={"photo": ("monture.jpg", _fichier_image_valide(), "image/jpeg")})
    assert len(r.json()["resultats"]) == 1


def test_route_identifier_k_invalide_rejete(client):
    r = client.post("/identifier?k=0", files={"photo": ("monture.jpg", _fichier_image_valide(), "image/jpeg")})
    assert r.status_code == 422  # k >= 1 imposé par la route


def test_route_lire_etiquette_ok(client):
    r = client.post("/lire-etiquette", files={"photo": ("etiquette.jpg", _fichier_image_valide(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["marque"] == "Ray-Ban"


def test_route_identifier_fichier_non_image_rejete(client):
    r = client.post("/identifier", files={"photo": ("notes.txt", b"ceci n'est pas une image", "text/plain")})
    assert r.status_code == 400


def test_route_identifier_sans_fichier_rejete(client):
    r = client.post("/identifier")
    assert r.status_code == 422  # champ 'photo' obligatoire
