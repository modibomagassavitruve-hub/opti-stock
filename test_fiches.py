"""La saisie au comptoir doit être permissive sur ce qu'elle accepte et stricte sur ce qu'elle
promet : ces tests cadrent les deux."""
import pytest

from fiches import Fiches


@pytest.fixture
def fiches(tmp_path):
    return Fiches(tmp_path / "fiches")


def test_creer_une_fiche_minimale(fiches):
    """Au comptoir, la référence n'est pas toujours lisible : exiger plus que la marque ferait
    abandonner la saisie."""
    f = fiches.creer({"marque": "OCTIKA"})
    assert f["monture"].startswith("f_")
    assert f["marque"] == "OCTIKA"
    assert f["reference"] == ""


def test_marque_obligatoire(fiches):
    for champs in ({}, {"marque": ""}, {"marque": "   "}, {"reference": "OS866"}):
        with pytest.raises(ValueError, match="marque"):
            fiches.creer(champs)


def test_identifiant_prefixe_pour_distinguer_du_catalogue(fiches):
    """Dans le stock et l'inventaire, on doit toujours savoir ce qui vient d'une saisie."""
    assert fiches.creer({"marque": "A"})["monture"].startswith("f_")


def test_identifiants_uniques(fiches):
    ids = {fiches.creer({"marque": "A"})["monture"] for _ in range(20)}
    assert len(ids) == 20


def test_tous_les_champs_conserves(fiches):
    f = fiches.creer({"marque": "RAY-BAN", "reference": "RB3025", "coloris": "001/51",
                      "calibre": "58", "pont": "14", "branche": "135"})
    assert (f["coloris"], f["calibre"], f["pont"], f["branche"]) == ("001/51", "58", "14", "135")


def test_espaces_rognes(fiches):
    assert fiches.creer({"marque": "  OCTIKA  ", "reference": " OS866 "})["marque"] == "OCTIKA"


# ------------------------------------------------------------------- photo
def test_photo_conservee(fiches):
    f = fiches.creer({"marque": "A"}, photo=b"\xff\xd8jpeg")
    assert f["photo"] is True
    assert fiches.photo(f["monture"]).read_bytes() == b"\xff\xd8jpeg"


def test_fiche_sans_photo(fiches):
    f = fiches.creer({"marque": "A"})
    assert f["photo"] is False
    assert fiches.photo(f["monture"]) is None


# --------------------------------------------------------------- corrections
def test_modifier_une_fiche(fiches):
    f = fiches.creer({"marque": "OCTKA"})            # faute de frappe au comptoir
    corrigee = fiches.modifier(f["monture"], {"marque": "OCTIKA", "reference": "OS866"})
    assert (corrigee["marque"], corrigee["reference"]) == ("OCTIKA", "OS866")


def test_modification_en_ajout_seul(fiches):
    """L'historique des corrections reste lisible, comme partout dans ce projet."""
    f = fiches.creer({"marque": "A"})
    avant = fiches.lignes.read_text()
    fiches.modifier(f["monture"], {"marque": "B"})
    assert fiches.lignes.read_text().startswith(avant)


def test_modifier_ne_touche_pas_aux_champs_absents(fiches):
    f = fiches.creer({"marque": "A", "reference": "R1", "calibre": "52"})
    corrigee = fiches.modifier(f["monture"], {"reference": "R2"})
    assert (corrigee["marque"], corrigee["calibre"]) == ("A", "52")


def test_modifier_une_fiche_inconnue(fiches):
    with pytest.raises(KeyError):
        fiches.modifier("f_inexistante", {"marque": "A"})


# ------------------------------------------------------------------ lecture
def test_libelle_marque_et_reference(fiches):
    f = fiches.creer({"marque": "OCTIKA", "reference": "OS866"})
    assert fiches.libelle(f["monture"]) == "OCTIKA OS866"


def test_libelle_sans_reference(fiches):
    f = fiches.creer({"marque": "OCTIKA"})
    assert fiches.libelle(f["monture"]) == "OCTIKA"


def test_libelle_d_un_identifiant_du_catalogue(fiches):
    """Une monture du catalogue entraîné n'a pas de fiche : on renvoie son identifiant tel
    quel plutôt que de planter."""
    assert fiches.libelle("18") == "18"


def test_chercher_par_marque_ou_reference(fiches):
    fiches.creer({"marque": "OCTIKA", "reference": "OS866"})
    fiches.creer({"marque": "RAY-BAN", "reference": "RB3025"})
    assert len(fiches.chercher("octika")) == 1
    assert len(fiches.chercher("RB30")) == 1
    assert len(fiches.chercher("")) == 2, "recherche vide : tout, pour parcourir"


def test_fiches_vides(fiches):
    assert fiches.toutes() == {}
    assert fiches.chercher("x") == []
    assert fiches.fiche("f_rien") is None
