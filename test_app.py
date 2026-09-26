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


def test_inventaire_bout_en_bout(client, tmp_path, monkeypatch):
    """Le parcours complet : le stock annonce un théorique, le rayon dit autre chose, et la
    clôture corrige le stock d'après le rayon."""
    from inventaire import Inventaire
    from stock import Stock
    monkeypatch.setattr(app_module, "INVENTAIRES", Inventaire(tmp_path / "inv"))
    stock = Stock(tmp_path / "stock.jsonl")
    monkeypatch.setattr(app_module, "STOCK", stock)
    stock.entrer("talla/bogart2", 3)
    stock.entrer("talla/gravita9015", 1)

    session = client.post("/inventaire?libelle=2026").json()
    assert session["pieces_attendues"] == 4
    sid = session["session"]

    etat = client.post(f"/inventaire/{sid}/compter?monture=talla/bogart2&quantite=2").json()
    assert etat["pieces_comptees"] == 2
    assert etat["manquantes"][0]["ecart"] == -1     # 2 comptées sur 3 attendues

    assert client.post(f"/inventaire/{sid}/cloturer").json()["statut"] == "clos"
    assert stock.quantite("talla/bogart2") == 2, "le rayon fait foi"
    assert stock.quantite("talla/gravita9015") == 0, "non comptée : elle n'est plus là"

    # un inventaire clos n'accepte plus de comptage : le décompte final ne doit plus bouger
    assert client.post(f"/inventaire/{sid}/compter?monture=talla/bogart2").status_code == 409
    assert client.post(f"/inventaire/{sid}/cloturer").status_code == 409


def test_inventaire_inconnu_renvoie_404(client, tmp_path, monkeypatch):
    from inventaire import Inventaire
    monkeypatch.setattr(app_module, "INVENTAIRES", Inventaire(tmp_path))
    assert client.get("/inventaire/zzz").status_code == 404
    assert client.post("/inventaire/zzz/compter?monture=a").status_code == 404


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


# ------------------------------------------------- routes réseau (module 3)
@pytest.fixture()
def reseau_vide(client, tmp_path, monkeypatch):
    """Réseau isolé par test. Sans cette redirection, les tests écriraient dans le vrai journal
    de la boutique : des inscriptions et des messages factices dans les données de production."""
    from reseau import Reseau
    from stock import Stock

    monkeypatch.setattr(app_module, "RESEAU", Reseau(tmp_path / "reseau.jsonl"))
    monkeypatch.setattr(app_module, "_references_catalogue",
                        lambda: {"m1": "OS866", "m2": "6135"})
    client.app.state.modeles = _modeles_avec_marques()

    # Le réseau ne propose que ce qui reste en rayon : sans stock, rien n'est partageable.
    stock = Stock(tmp_path / "stock.jsonl")
    monkeypatch.setattr(app_module, "STOCK", stock)
    for monture in ("m1", "m2", "m3"):
        stock.entrer(monture, 1)
    return client


@pytest.fixture()
def reseau(reseau_vide):
    """Le réseau isolé, avec deux boutiques déjà inscrites."""
    client = reseau_vide
    a = client.post("/reseau/inscription?nom=Optique Centre&ville=Lyon").json()
    b = client.post("/reseau/inscription?nom=Vision Plus&ville=Villeurbanne").json()
    return client, {"X-Boutique-Jeton": a["jeton"]}, {"X-Boutique-Jeton": b["jeton"]}, a, b


def test_inscription_renvoie_un_jeton(reseau_vide):
    r = reseau_vide.post("/reseau/inscription?nom=Optique Centre&ville=Lyon")
    assert r.status_code == 200
    assert r.json()["jeton"] and r.json()["boutique"]


def test_inscription_refuse_un_nom_vide(reseau_vide):
    assert reseau_vide.post("/reseau/inscription?nom=A").status_code == 422


def test_annuaire_n_expose_pas_les_jetons(reseau):
    client, *_ = reseau
    fiches = client.get("/reseau/boutiques").json()
    assert [f["nom"] for f in fiches] == ["Optique Centre", "Vision Plus"]
    assert all("jeton" not in f and "email" not in f for f in fiches)


def test_routes_reseau_exigent_un_jeton(reseau):
    """Sans jeton valide, aucune route du réseau ne répond."""
    client, entete_a, *_ = reseau
    protegees = [("get", "/reseau/mon-stock"), ("get", "/reseau/chercher?marque=Osmose"),
                 ("get", "/reseau/conversations"), ("post", "/reseau/partage"),
                 ("post", "/reseau/demande?boutique_sollicitee=x&texte=bonjour")]
    for methode, url in protegees:
        assert getattr(client, methode)(url).status_code == 401, url
        assert getattr(client, methode)(
            url, headers={"X-Boutique-Jeton": "faux"}).status_code == 401, url


def test_mon_stock_ne_propose_que_les_montures_a_marque(reseau):
    client, entete_a, *_ = reseau
    montures = client.get("/reseau/mon-stock", headers=entete_a).json()["montures"]
    assert {m["libelle"] for m in montures} == {"m1", "m2", "m3"}
    assert all(m["marque"] for m in montures)
    assert not any(m["partagee"] for m in montures)


def test_partage_puis_recherche_par_un_confrere(reseau):
    """Le parcours de la démonstration, en HTTP."""
    client, entete_a, entete_b, _, b = reseau
    assert client.post("/reseau/partage?montures=m1", headers=entete_b).json()["partagees"] == 1

    trouves = client.get("/reseau/chercher?marque=Osmose&reference=OS866",
                          headers=entete_a).json()
    assert len(trouves) == 1
    assert trouves[0]["nom"] == "Vision Plus"
    assert trouves[0]["boutique"] == b["boutique"]


def test_partage_refuse_une_monture_absente_du_stock(reseau):
    client, entete_a, *_ = reseau
    r = client.post("/reseau/partage?montures=inexistante", headers=entete_a)
    assert r.status_code == 400


def test_partage_vide_retire_tout(reseau):
    client, entete_a, entete_b, *_ = reseau
    client.post("/reseau/partage?montures=m1", headers=entete_b)
    client.post("/reseau/partage", headers=entete_b)
    assert client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json() == []


def test_recherche_sans_critere_rejetee(reseau):
    client, entete_a, *_ = reseau
    assert client.get("/reseau/chercher", headers=entete_a).status_code == 400


def test_demande_puis_reponse_de_bout_en_bout(reseau):
    client, entete_a, entete_b, _, b = reseau
    client.post("/reseau/partage?montures=m1", headers=entete_b)

    conv = client.post(f"/reseau/demande?boutique_sollicitee={b['boutique']}"
                        "&monture=Osmose OS866&texte=Encore disponible ?",
                        headers=entete_a).json()["conversation"]

    # B voit la demande arriver sans avoir eu besoin d'un identifiant transmis hors du produit
    fils = client.get("/reseau/conversations", headers=entete_b).json()
    assert fils[0]["interlocuteur"] == "Optique Centre"
    assert fils[0]["monture"] == "Osmose OS866"

    client.post(f"/reseau/conversation/{conv}/message?texte=Oui, je la mets de côté",
                headers=entete_b)
    fil = client.get(f"/reseau/conversation/{conv}", headers=entete_a).json()
    assert [m["nom"] for m in fil["messages"]] == ["Optique Centre", "Vision Plus"]


def test_une_boutique_tierce_ne_lit_pas_la_conversation(reseau):
    """L'étanchéité, vérifiée au niveau HTTP et pas seulement dans le module."""
    client, entete_a, entete_b, _, b = reseau
    conv = client.post(f"/reseau/demande?boutique_sollicitee={b['boutique']}"
                        "&texte=Prix confidentiel 40 EUR", headers=entete_a).json()["conversation"]

    c = client.post("/reseau/inscription?nom=Tiers Curieux&ville=Paris").json()
    entete_c = {"X-Boutique-Jeton": c["jeton"]}
    assert client.get(f"/reseau/conversation/{conv}", headers=entete_c).status_code == 403
    assert client.post(f"/reseau/conversation/{conv}/message?texte=coucou",
                        headers=entete_c).status_code == 403
    assert client.get("/reseau/conversations", headers=entete_c).json() == []


def test_demande_a_une_boutique_inconnue(reseau):
    client, entete_a, *_ = reseau
    r = client.post("/reseau/demande?boutique_sollicitee=00000000&texte=bonjour",
                     headers=entete_a)
    assert r.status_code == 404


def test_conversation_inconnue_donne_404(reseau):
    client, entete_a, *_ = reseau
    assert client.get("/reseau/conversation/inexistante", headers=entete_a).status_code == 404


def test_blocage_coupe_le_contact_et_masque_le_stock(reseau):
    client, entete_a, entete_b, a, b = reseau
    client.post("/reseau/partage?montures=m1", headers=entete_b)
    client.post(f"/reseau/bloquer?boutique_bloquee={a['boutique']}", headers=entete_b)

    assert client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json() == []
    r = client.post(f"/reseau/demande?boutique_sollicitee={b['boutique']}&texte=bonjour",
                     headers=entete_a)
    assert r.status_code == 403


def test_signalement_reserve_aux_participants(reseau):
    client, entete_a, entete_b, _, b = reseau
    conv = client.post(f"/reseau/demande?boutique_sollicitee={b['boutique']}&texte=bonjour",
                        headers=entete_a).json()["conversation"]

    c = client.post("/reseau/inscription?nom=Tiers&ville=Paris").json()
    assert client.post(f"/reseau/signaler?conversation={conv}&motif=spam",
                        headers={"X-Boutique-Jeton": c["jeton"]}).status_code == 403
    assert client.post(f"/reseau/signaler?conversation={conv}&motif=spam",
                        headers=entete_b).status_code == 200


def test_mon_stock_liste_une_monture_par_entree_pas_une_par_photo(reseau):
    """Le catalogue compte plusieurs photos par monture : les lister toutes rendrait le choix
    des montures à partager illisible (constaté : chaque monture affichée cinq fois)."""
    client, entete_a, *_ = reseau
    client.app.state.modeles.labels_catalogue = np.array(["m1", "m1", "m1", "m2"])
    client.app.state.modeles.marques_catalogue = np.array(["Osmose"] * 3 + ["Maritza"])

    montures = client.get("/reseau/mon-stock", headers=entete_a).json()["montures"]
    assert [m["libelle"] for m in montures] == ["m2", "m1"]  # triées par marque


# ------------------------------------------------- saisie des marques
@pytest.fixture()
def saisie(client, tmp_path, monkeypatch):
    """CSV isolé : sans ça les tests écriraient dans le vrai fichier de stock."""
    import csv as _csv

    import metadonnees as md
    chemin = tmp_path / "montures.csv"
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        (w := _csv.DictWriter(f, md.CHAMPS)).writeheader()
        w.writerows([
            {"monture": "m1", "marque": "OSMOSE", "reference": "", "ean": "", "etat": "saisi"},
            {"monture": "m2", "marque": "", "reference": "", "ean": "", "etat": "a_saisir"},
            {"monture": "m3", "marque": "NEMEZIS", "reference": "", "ean": "", "etat": "ocr_a_verifier"},
        ])
    monkeypatch.setattr(app_module, "METADONNEES", chemin)
    return client, chemin


def test_saisie_liste_ce_qui_reste(saisie):
    client, _ = saisie
    d = client.get("/saisie/montures").json()
    assert {m["monture"] for m in d["montures"]} == {"m2", "m3"}
    assert d["bilan"]["sans_marque"] == 1
    assert d["bilan"]["a_verifier"] == 1


def test_saisie_enregistre_et_met_a_jour_le_bilan(saisie):
    client, _ = saisie
    r = client.post("/saisie/marques", json={"m2": {"marque": "IKALY", "reference": "IK1"}})
    assert r.status_code == 200
    assert r.json()["modifiees"] == 1
    assert r.json()["bilan"]["sans_marque"] == 0
    assert "IKALY" in r.json()["bilan"]["marques"]


def test_saisie_corriger_une_lecture_ocr_la_retire_de_la_liste(saisie):
    """Retaper la marque est un geste humain : pas besoin de cocher en plus."""
    client, _ = saisie
    client.post("/saisie/marques", json={"m3": {"marque": "IKALY"}})
    assert {m["monture"] for m in client.get("/saisie/montures").json()["montures"]} == {"m2"}


def test_saisie_monture_inconnue_rejetee_sans_rien_ecrire(saisie):
    client, chemin = saisie
    avant = chemin.read_text()
    r = client.post("/saisie/marques", json={"inexistante": {"marque": "X"}})
    assert r.status_code == 400
    assert chemin.read_text() == avant


def test_saisie_marque_vide_efface(saisie):
    """Corriger une lecture OCR fausse en effaçant le champ."""
    client, _ = saisie
    client.post("/saisie/marques", json={"m3": {"marque": ""}})
    assert client.get("/saisie/montures").json()["bilan"]["sans_marque"] == 2


def test_page_saisie_servie(saisie):
    client, _ = saisie
    r = client.get("/saisie")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_saisie_enregistrer_ne_confirme_pas_les_lectures_ocr(saisie):
    """Enregistrer la page ne doit pas valider une marque devinée que personne n'a relue :
    elle resterait sinon proposée au réseau comme un fait établi."""
    client, _ = saisie
    client.post("/saisie/marques", json={"m3": {"marque": "NEMEZIS"}})
    restantes = {m["monture"]: m for m in client.get("/saisie/montures").json()["montures"]}
    assert restantes["m3"]["etat"] == "ocr_a_verifier"


def test_saisie_confirmation_explicite_valide(saisie):
    client, _ = saisie
    client.post("/saisie/marques", json={"m3": {"marque": "NEMEZIS", "confirmee": True}})
    assert {m["monture"] for m in client.get("/saisie/montures").json()["montures"]} == {"m2"}


# ------------------------------------------------- stock (module 1)
@pytest.fixture()
def stock_api(client, tmp_path, monkeypatch):
    from stock import Stock
    s = Stock(tmp_path / "stock.jsonl")
    monkeypatch.setattr(app_module, "STOCK", s)
    client.app.state.modeles = _modeles_avec_marques()
    return client, s


def test_entree_en_stock_puis_lecture(stock_api):
    client, _ = stock_api
    r = client.post("/stock/m1/entree?quantite=3&emplacement=vitrine A")
    assert r.status_code == 200
    assert r.json()["quantite"] == 3

    d = client.get("/stock").json()
    ligne = [l for l in d["lignes"] if l["monture"] == "m1"][0]
    assert (ligne["quantite"], ligne["emplacement"]) == (3, "vitrine A")
    assert ligne["marque"] == "Osmose", "la marque du catalogue rend le stock lisible"
    assert d["bilan"]["pieces"] == 3


def test_vente_diminue_le_stock(stock_api):
    client, _ = stock_api
    client.post("/stock/m1/entree?quantite=2")
    assert client.post("/stock/m1/sortie?quantite=1&motif=vente").json()["quantite"] == 1


def test_vendre_plus_que_le_stock_renvoie_409(stock_api):
    """Un stock négatif n'existe pas en rayon : le refus doit être franc, pas silencieux."""
    client, s = stock_api
    client.post("/stock/m1/entree?quantite=1")
    assert client.post("/stock/m1/sortie?quantite=5").status_code == 409
    assert s.quantite("m1") == 1


def test_motif_invalide_refuse(stock_api):
    client, _ = stock_api
    assert client.post("/stock/m1/entree?motif=evaporation").status_code == 400


def test_ajustement_et_historique(stock_api):
    client, _ = stock_api
    client.post("/stock/m1/entree?quantite=5")
    client.post("/stock/m1/ajuster?quantite=2")
    d = client.get("/stock/m1").json()
    assert d["quantite"] == 2
    assert [m["type"] for m in d["mouvements"]] == ["entree", "ajustement"]
    assert d["mouvements"][-1]["ecart"] == -3


def test_stock_vide(stock_api):
    client, _ = stock_api
    assert client.get("/stock").json()["bilan"]["pieces"] == 0
    assert client.get("/stock/jamais_vue").json()["quantite"] == 0


# ------------------------------------------------- stock x réseau
def test_le_reseau_ne_propose_que_ce_qui_reste_en_rayon(reseau):
    """Le défaut que ce module corrige : proposer une monture vendue fait déplacer un confrère
    pour rien."""
    client, entete_a, entete_b, _, b = reseau
    client.post("/reseau/partage?montures=m1", headers=entete_b)
    assert len(client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json()) == 1

    client.post("/stock/m1/sortie?quantite=1&motif=vente")
    assert client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json() == []


def test_monture_epuisee_nest_plus_partageable(reseau):
    client, entete_a, _, _, _ = reseau
    client.post("/stock/m1/sortie?quantite=1&motif=vente")
    libelles = {m["libelle"] for m in
                client.get("/reseau/mon-stock", headers=entete_a).json()["montures"]}
    assert "m1" not in libelles


def test_le_partage_porte_la_quantite_disponible(reseau):
    client, entete_a, entete_b, _, _ = reseau
    client.post("/stock/m2/entree?quantite=4")
    client.post("/reseau/partage?montures=m2", headers=entete_b)
    trouve = client.get("/reseau/chercher?marque=Maritza", headers=entete_a).json()[0]
    assert trouve["quantite"] == 5


def test_la_disponibilite_est_verifiee_a_la_recherche_pas_au_partage(reseau):
    """Le partage est figé dans le journal ; le stock bouge. Une boutique partage le matin,
    vend à midi : l'après-midi la monture ne doit plus apparaître, sans qu'elle ait eu à
    repartager quoi que ce soit."""
    client, entete_a, entete_b, _, _ = reseau
    client.post("/reseau/partage?montures=m1", headers=entete_b)
    client.post("/stock/m1/sortie?quantite=1&motif=vente")

    assert client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json() == []
    # puis elle en reçoit à nouveau : elle réapparaît sans nouveau partage
    client.post("/stock/m1/entree?quantite=2")
    trouve = client.get("/reseau/chercher?marque=Osmose", headers=entete_a).json()
    assert len(trouve) == 1 and trouve[0]["quantite"] == 2


# ------------------------------------------------- boucle d'apprentissage
def test_route_seuil_est_en_lecture_seule(client, tmp_path, monkeypatch):
    """Recalibrer modifie ce que l'API affirme : ça passe par la ligne de commande, jamais par
    une requête web."""
    from journal import Journal
    monkeypatch.setattr(app_module, "JOURNAL", Journal(tmp_path / "journal"))
    avant = client.get("/journal/seuil").json()
    assert avant["validations"] == 0 and avant["assez"] is False
    assert client.post("/journal/seuil").status_code == 405


def test_route_seuil_mesure_le_seuil_en_service(client, tmp_path, monkeypatch):
    import json as _json

    from journal import Journal
    journal = Journal(tmp_path / "journal")
    monkeypatch.setattr(app_module, "JOURNAL", journal)
    tete = tmp_path / "tete.json"
    tete.write_text(_json.dumps({"seuil_confiance": 0.5}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tete.json").write_text(_json.dumps({"seuil_confiance": 0.5}),
                                                  encoding="utf-8")

    for monture, choix in (("a", "a"), ("b", "aucune")):
        ident = journal.enregistrer(
            type_prediction="similarite", version="t",
            resultat={"resultats": [{"monture": monture, "similarite": 0.9}], "fiable": True})
        journal.noter_choix(ident, choix)

    d = client.get("/journal/seuil").json()
    assert d["validations"] == 2
    assert d["reel_au_seuil_actuel"]["precision"] == 0.5
