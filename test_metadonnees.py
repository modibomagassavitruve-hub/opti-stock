"""Ce module réécrit le fichier de stock en entier : les tests portent surtout sur ce qu'il ne
doit pas abîmer."""
import csv

import pytest

import metadonnees as md


@pytest.fixture
def fichier(tmp_path):
    chemin = tmp_path / "montures.csv"
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        (w := csv.DictWriter(f, md.CHAMPS)).writeheader()
        w.writerows([
            {"monture": "1", "marque": "OSMOSE", "reference": "OS866", "ean": "3700", "etat": "saisi"},
            {"monture": "2", "marque": "", "reference": "", "ean": "", "etat": "a_saisir"},
            {"monture": "3", "marque": "NEMEZIS", "reference": "", "ean": "", "etat": "ocr_a_verifier"},
        ])
    return chemin


def test_lire_rend_toutes_les_colonnes(fichier):
    lignes = md.lire(fichier)
    assert len(lignes) == 3
    assert lignes[0] == {"monture": "1", "marque": "OSMOSE", "reference": "OS866",
                          "ean": "3700", "etat": "saisi"}


def test_fichier_absent_ne_plante_pas(tmp_path):
    assert md.lire(tmp_path / "rien.csv") == []
    assert md.a_completer(tmp_path / "rien.csv") == []


def test_a_completer_prend_les_vides_et_les_a_verifier(fichier):
    """Une marque déduite de l'OCR doit repasser devant l'opticien, même si elle est remplie."""
    assert {l["monture"] for l in md.a_completer(fichier)} == {"2", "3"}


def test_ecrire_renseigne_la_marque(fichier):
    assert md.ecrire_marques(fichier, {"2": "IKALY"}) == 1
    ligne = [l for l in md.lire(fichier) if l["monture"] == "2"][0]
    assert ligne["marque"] == "IKALY"
    assert ligne["etat"] == md.ETAT_SAISIE


def test_ecrire_ne_touche_pas_aux_autres_lignes(fichier):
    """Le fichier est réécrit en entier : le risque est d'y perdre ce qu'on ne visait pas."""
    avant = md.lire(fichier)
    md.ecrire_marques(fichier, {"2": "IKALY"})
    apres = {l["monture"]: l for l in md.lire(fichier)}
    for ligne in avant:
        if ligne["monture"] != "2":
            assert apres[ligne["monture"]] == ligne


def test_ean_preserve(fichier):
    """L'EAN n'est jamais saisi par cette page : il doit traverser l'écriture intact."""
    md.ecrire_marques(fichier, {"1": "OSMOSE BIS"})
    assert [l for l in md.lire(fichier) if l["monture"] == "1"][0]["ean"] == "3700"


def test_ecrire_la_reference_aussi(fichier):
    md.ecrire_marques(fichier, {"2": "IKALY"}, {"2": "IK123"})
    ligne = [l for l in md.lire(fichier) if l["monture"] == "2"][0]
    assert (ligne["marque"], ligne["reference"]) == ("IKALY", "IK123")


def test_marque_vide_efface_et_repasse_a_saisir(fichier):
    """C'est ainsi qu'on corrige une lecture OCR fausse."""
    assert md.ecrire_marques(fichier, {"3": ""}) == 1
    ligne = [l for l in md.lire(fichier) if l["monture"] == "3"][0]
    assert ligne["marque"] == ""
    assert ligne["etat"] == "a_saisir"


def test_completer_la_reference_ne_valide_pas_la_marque(fichier):
    """Saisir la référence sans toucher à la marque laisse celle-ci à l'état de lecture OCR :
    elle n'a toujours pas été relue."""
    assert md.ecrire_marques(fichier, {"3": "NEMEZIS"}, {"3": "DAMASENE"}) == 1
    ligne = [l for l in md.lire(fichier) if l["monture"] == "3"][0]
    assert (ligne["reference"], ligne["etat"]) == ("DAMASENE", "ocr_a_verifier")
    assert {l["monture"] for l in md.a_completer(fichier)} == {"2", "3"}


def test_monture_inconnue_refusee_sans_rien_ecrire(fichier):
    avant = fichier.read_text()
    with pytest.raises(KeyError, match="99"):
        md.ecrire_marques(fichier, {"2": "IKALY", "99": "FANTOME"})
    assert fichier.read_text() == avant, "un refus ne doit rien modifier"


def test_renvoyer_une_lecture_ocr_telle_quelle_ne_la_confirme_pas(fichier):
    """Enregistrer une page entière ne doit pas valider les lectures OCR qu'elle contient :
    sans relecture humaine, la ligne reste à vérifier."""
    assert md.ecrire_marques(fichier, {"3": "NEMEZIS"}) == 0
    assert [l for l in md.lire(fichier) if l["monture"] == "3"][0]["etat"] == "ocr_a_verifier"


def test_confirmation_explicite_valide_la_ligne(fichier):
    assert md.ecrire_marques(fichier, {"3": "NEMEZIS"}, confirmees={"3"}) == 1
    assert [l for l in md.lire(fichier) if l["monture"] == "3"][0]["etat"] == md.ETAT_SAISIE


def test_corriger_une_lecture_ocr_vaut_confirmation(fichier):
    """Retaper la marque est un geste humain : pas besoin de cocher en plus."""
    md.ecrire_marques(fichier, {"3": "IKALY"})
    ligne = [l for l in md.lire(fichier) if l["monture"] == "3"][0]
    assert (ligne["marque"], ligne["etat"]) == ("IKALY", md.ETAT_SAISIE)


def test_ligne_deja_saisie_et_inchangee_nest_pas_reecrite(fichier):
    assert md.ecrire_marques(fichier, {"1": "OSMOSE"}) == 0


def test_les_espaces_sont_rognes(fichier):
    md.ecrire_marques(fichier, {"2": "  IKALY  "})
    assert [l for l in md.lire(fichier) if l["monture"] == "2"][0]["marque"] == "IKALY"


def test_bilan(fichier):
    b = md.bilan(fichier)
    assert (b["montures"], b["avec_marque"], b["sans_marque"], b["a_verifier"]) == (3, 2, 1, 1)
    assert b["marques"] == ["NEMEZIS", "OSMOSE"]


def test_ecriture_atomique_laisse_un_csv_lisible(fichier, monkeypatch):
    """Une coupure pendant l'écriture ne doit pas laisser un stock tronqué à la place."""
    monkeypatch.setattr(md.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("coupure")))
    with pytest.raises(OSError):
        md.ecrire_marques(fichier, {"2": "IKALY"})
    assert len(md.lire(fichier)) == 3  # l'original est intact
