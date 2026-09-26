"""Ce module peut modifier le seuil servi et le jeu d'entraînement : les tests portent d'abord
sur ses refus, puis sur la distinction « aucune » qui traverse tout le module."""
import json

import pytest

import apprendre
from journal import Journal


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal")


@pytest.fixture
def tete_json(tmp_path):
    chemin = tmp_path / "tete.json"
    chemin.write_text(json.dumps({"seuil_confiance": 0.777, "backbone": "fashionclip"}),
                      encoding="utf-8")
    return chemin


def _predire(journal, monture_proposee, similarite, choix=None, photo=b"\xff\xd8jpeg"):
    """Une identification, éventuellement tranchée par l'opticien."""
    ident = journal.enregistrer(
        type_prediction="similarite", version="test", photo=photo,
        resultat={"resultats": [{"monture": monture_proposee, "similarite": similarite}],
                  "fiable": similarite >= 0.777})
    if choix is not None:
        journal.noter_choix(ident, choix)
    return ident


# ------------------------------------------------------------------ cas valides
def test_seule_une_prediction_tranchee_compte(journal):
    _predire(journal, "a", 0.9)                 # jamais validée
    _predire(journal, "b", 0.9, choix="b")
    assert [c["monture_choisie"] for c in apprendre.cas_valides(journal)] == ["b"]


def test_aucune_correspondance_compte_comme_fausse(journal):
    """« Aucune ne correspond » dit que la première proposition était fausse. C'est une donnée,
    et l'une des plus utiles pour régler un seuil."""
    _predire(journal, "a", 0.95, choix="aucune")
    cas = apprendre.cas_valides(journal)
    assert len(cas) == 1
    assert cas[0]["juste"] is False
    assert cas[0]["etiquetee"] is False


def test_choix_different_de_la_proposition_est_faux(journal):
    _predire(journal, "a", 0.9, choix="b")
    assert apprendre.cas_valides(journal)[0]["juste"] is False


def test_journal_vide(journal, tete_json):
    assert apprendre.cas_valides(journal) == []
    rapport = apprendre.seuil_reel(journal, tete_json)
    assert rapport["validations"] == 0 and rapport["assez"] is False


# --------------------------------------------------------------------- refus
def test_pas_assez_de_validations_ne_propose_rien(journal, tete_json):
    """Sur une poignée de cas, un seuil recalculé viendrait du bruit et serait pire que celui
    en place."""
    for i in range(5):
        _predire(journal, "a", 0.9, choix="a")
    rapport = apprendre.seuil_reel(journal, tete_json)
    assert rapport["assez"] is False
    assert "seuil_propose" not in rapport


def test_le_seuil_en_service_est_mesure_meme_sans_assez_de_cas(journal, tete_json):
    """Le chiffre doit être visible tôt : c'est lui qui alerte si le seuil est trop permissif."""
    _predire(journal, "a", 0.95, choix="aucune")
    _predire(journal, "b", 0.95, choix="b")
    reel = apprendre.seuil_reel(journal, tete_json)["reel_au_seuil_actuel"]
    assert reel["repondus"] == 2
    assert reel["precision"] == 0.5


def test_appliquer_refuse_sans_assez_de_validations(journal, tete_json, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["apprendre.py", "--journal", str(journal.dossier),
                                      "--tete-json", str(tete_json), "--appliquer-seuil"])
    _predire(journal, "a", 0.9, choix="a")
    with pytest.raises(SystemExit):
        apprendre.main()
    assert json.loads(tete_json.read_text())["seuil_confiance"] == 0.777, "seuil inchangé"


def test_seuil_inatteignable_ne_propose_rien(journal, tete_json):
    """Si rien n'atteint la précision visée, l'API ne doit jamais se déclarer fiable."""
    for i in range(40):
        _predire(journal, "a", 0.5 + i / 100, choix="aucune")   # toutes fausses
    rapport = apprendre.seuil_reel(journal, tete_json)
    assert rapport["assez"] is True
    assert rapport["atteignable"] is False


# --------------------------------------------------------- recalibration
def test_recalibration_sur_donnees_reelles(journal, tete_json):
    """Les cas justes ont des scores hauts, les faux des scores bas : le seuil proposé doit se
    placer entre les deux."""
    for _ in range(25):
        _predire(journal, "a", 0.95, choix="a")
    for _ in range(15):
        _predire(journal, "b", 0.60, choix="aucune")

    rapport = apprendre.seuil_reel(journal, tete_json)
    assert rapport["assez"] and rapport["atteignable"]
    assert 0.60 < rapport["seuil_propose"] <= 0.95
    assert rapport["reel_au_seuil_propose"]["precision"] >= 0.90


def test_appliquer_conserve_le_seuil_precedent(tete_json):
    """Pouvoir revenir en arrière : une recalibration sur un usage atypique doit se défaire."""
    apprendre.appliquer_seuil(tete_json, 0.83)
    config = json.loads(tete_json.read_text())
    assert config["seuil_confiance"] == 0.83
    assert config["seuil_precedent"] == 0.777
    assert config["backbone"] == "fashionclip", "le reste du fichier est préservé"


# ------------------------------------------------------------------- photos
def test_photos_versees_dans_le_dossier_de_leur_monture(journal, tmp_path):
    racine = tmp_path / "mes_montures"
    _predire(journal, "a", 0.9, choix="a")
    verses = apprendre.verser_photos(journal, racine)
    assert len(verses) == 1
    assert len(list((racine / "a").glob("journal_*.jpg"))) == 1


def test_aucune_correspondance_ne_donne_pas_de_photo(journal, tmp_path):
    """Elle dit que la proposition était fausse, pas quelle était la bonne monture : sans
    étiquette, pas d'exemple d'entraînement."""
    racine = tmp_path / "mes_montures"
    _predire(journal, "a", 0.9, choix="aucune")
    assert apprendre.photos_a_verser(journal, racine) == []
    assert apprendre.verser_photos(journal, racine) == []


def test_le_versement_ne_duplique_pas(journal, tmp_path):
    racine = tmp_path / "mes_montures"
    _predire(journal, "a", 0.9, choix="a")
    apprendre.verser_photos(journal, racine)
    assert apprendre.verser_photos(journal, racine) == [], "deuxième passage : rien à verser"
    assert len(list((racine / "a").glob("journal_*.jpg"))) == 1


def test_supprimer_une_photo_versee_la_rend_a_verser(journal, tmp_path):
    """Le versement se défait en supprimant le fichier : pas de registre séparé à resynchroniser
    quand l'opticien a cliqué à côté."""
    racine = tmp_path / "mes_montures"
    _predire(journal, "a", 0.9, choix="a")
    apprendre.verser_photos(journal, racine)
    next((racine / "a").glob("journal_*.jpg")).unlink()
    assert len(apprendre.photos_a_verser(journal, racine)) == 1


def test_prediction_sans_photo_ignoree(journal, tmp_path):
    _predire(journal, "a", 0.9, choix="a", photo=None)
    assert apprendre.photos_a_verser(journal, tmp_path / "mes_montures") == []


def test_photo_versee_sous_le_nom_de_la_monture_choisie_pas_de_la_proposee(journal, tmp_path):
    """L'humain fait foi : si le modèle proposait « a » et que l'opticien a corrigé en « b »,
    la photo est un exemple de « b »."""
    racine = tmp_path / "mes_montures"
    _predire(journal, "a", 0.9, choix="b")
    apprendre.verser_photos(journal, racine)
    assert (racine / "b").is_dir() and not (racine / "a").exists()


# --------------------------------------------------------------------- rapport
def test_rapport_lisible_sans_donnees(journal, tete_json):
    texte = apprendre.rapport_texte(apprendre.seuil_reel(journal, tete_json), [])
    assert "Validations exploitables : 0" in texte
    assert "au moins 30" in texte
