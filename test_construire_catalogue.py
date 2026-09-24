import numpy as np

import construire_catalogue as cc


def _faux_backbone(monkeypatch, labels_par_chemin):
    """Remplace le backbone par un encodeur factice : un embedding one-hot par classe, dérivé du
    nom de fichier. Permet de vérifier que labels et embeddings restent alignés, sans charger de
    vrai modèle ni lire de vraies images."""
    classes = sorted(set(labels_par_chemin.values()))
    ouvertes = []

    def faux_ouvrir(chemin):
        ouvertes.append(str(chemin))
        return str(chemin)  # l'"image" est son propre chemin : l'encodeur factice s'en sert

    def faux_charger(nom, device):
        def encoder(images):
            base = np.eye(len(classes), dtype="float32")
            return np.array([base[classes.index(labels_par_chemin[i])] for i in images])
        return encoder

    monkeypatch.setattr(cc, "ouvrir_image", faux_ouvrir)
    monkeypatch.setattr(cc, "charger_backbone", faux_charger)
    monkeypatch.setattr(cc, "choisir_device", lambda: "cpu")
    return ouvertes


def _creer_montures(tmp_path, montures: dict[str, int]):
    """Crée data/mes_montures/<id>/photoN.jpg (fichiers vides, jamais ouverts par les tests
    puisque calculer_embeddings est mocké) sous tmp_path. Retourne racine + label par chemin."""
    racine = tmp_path / "mes_montures"
    par_chemin = {}
    for identifiant, n_photos in montures.items():
        dossier = racine / identifiant
        dossier.mkdir(parents=True)
        for i in range(n_photos):
            chemin = dossier / f"photo{i}.jpg"
            chemin.write_bytes(b"")
            par_chemin[str(chemin)] = identifiant
    return racine, par_chemin


def test_metadonnees_associent_la_marque_a_chaque_monture(tmp_path):
    csv = tmp_path / "montures.csv"
    csv.write_text("monture,marque\nm1,Osmose\nm2,Maritza\nm_absente,Talla\n", encoding="utf-8")
    assert cc.lire_metadonnees(csv, {"m1", "m2", "m3"}) == {"m1": "Osmose", "m2": "Maritza", "m3": ""}


def test_metadonnees_absentes_laissent_les_marques_vides(tmp_path):
    assert cc.lire_metadonnees(None, {"m1"}) == {"m1": ""}
    assert cc.lire_metadonnees(tmp_path / "inexistant.csv", {"m1"}) == {"m1": ""}


def test_catalogue_enregistre_les_marques(tmp_path, monkeypatch):
    racine, par_chemin = _creer_montures(tmp_path, {"m1": 2, "m2": 1})
    _faux_backbone(monkeypatch, par_chemin)
    csv = tmp_path / "montures.csv"
    csv.write_text("monture,marque\nm1,Osmose\n", encoding="utf-8")

    sortie = tmp_path / "catalogue.npz"
    cc.construire(racine, sortie, sans_recadrage=True, metadonnees=csv)

    d = np.load(sortie, allow_pickle=False)
    assert {l: m for l, m in zip(d["labels"], d["marques"])} == {"m1": "Osmose", "m2": ""}


def test_racine_vide_ne_produit_rien(tmp_path):
    racine = tmp_path / "mes_montures"
    racine.mkdir()
    resume = cc.construire(racine, tmp_path / "catalogue.npz", sans_recadrage=True)
    assert resume is None
    assert not (tmp_path / "catalogue.npz").exists()


def test_catalogue_contient_les_bonnes_cles_et_labels(tmp_path, monkeypatch):
    racine, par_chemin = _creer_montures(tmp_path, {"talla_bogart2": 2, "visionario_mikel03": 3})
    _faux_backbone(monkeypatch, par_chemin)

    sortie = tmp_path / "catalogue.npz"
    resume = cc.construire(racine, sortie, sans_recadrage=True)

    assert resume == {"n_photos": 5, "n_montures": 2, "sortie": sortie}
    assert sortie.exists()

    d = np.load(sortie, allow_pickle=False)
    assert set(d.files) >= {"emb", "labels", "chemins", "backbone"}
    assert str(d["backbone"]) == cc.BACKBONE_PROD  # le catalogue trace son backbone
    assert d["emb"].shape == (5, 2)  # 5 photos, 2 classes (embeddings one-hot factices)
    assert set(d["labels"]) == {"talla_bogart2", "visionario_mikel03"}
    assert (d["labels"] == "talla_bogart2").sum() == 2
    assert (d["labels"] == "visionario_mikel03").sum() == 3


def test_recadrage_appele_quand_pas_de_sans_recadrage(tmp_path, monkeypatch):
    racine, par_chemin = _creer_montures(tmp_path, {"m1": 2})
    _faux_backbone(monkeypatch, par_chemin)

    appels = {}

    def fausse_recadrer(chemins, racine_appel, dossier_cache):
        appels["chemins"] = chemins
        appels["dossier_cache"] = dossier_cache
        return chemins  # pas de vrai recadrage : renvoie les chemins tels quels

    monkeypatch.setattr(cc, "recadrer_dossier", fausse_recadrer)

    cc.construire(racine, tmp_path / "out.npz", sans_recadrage=False)

    assert appels["chemins"]  # bien appelé avec la liste de photos
    assert appels["dossier_cache"] == racine.parent / "mes_montures_crops"


def test_identifiant_unique_par_dossier_pas_par_marque(tmp_path, monkeypatch):
    """Le label est le nom du dossier tel quel (pas de découpage marque/modèle) : deux montures
    de la même marque doivent rester deux labels distincts."""
    racine, par_chemin = _creer_montures(tmp_path, {"talla_bogart2": 1, "talla_gravita9015": 1})
    _faux_backbone(monkeypatch, par_chemin)

    sortie = tmp_path / "out.npz"
    cc.construire(racine, sortie, sans_recadrage=True)

    d = np.load(sortie, allow_pickle=False)
    assert set(d["labels"]) == {"talla_bogart2", "talla_gravita9015"}
