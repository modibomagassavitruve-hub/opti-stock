import pytest

from inventaire import Inventaire


def test_session_fige_la_liste_attendue(tmp_path):
    """Le catalogue peut changer pendant l'inventaire sans fausser le décompte."""
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a", "b", "c"], libelle="2026")
    etat = inv.etat(s)
    assert etat["attendues"] == 3
    assert etat["statut"] == "en_cours"
    assert etat["libelle"] == "2026"


def test_ce_qui_compte_est_ce_quon_na_pas_trouve(tmp_path):
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a", "b", "c", "d"])
    inv.compter(s, "a")
    etat = inv.compter(s, "c")
    assert etat["trouvees"] == 2
    assert etat["manquantes"] == ["b", "d"]


def test_monture_absente_du_catalogue_signalee(tmp_path):
    """Une monture en rayon mais pas au catalogue est une anomalie à remonter, pas une erreur."""
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a", "b"])
    etat = inv.compter(s, "inconnue")
    assert etat["hors_catalogue"] == ["inconnue"]
    assert etat["trouvees"] == 0


def test_double_comptage_signale(tmp_path):
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a", "b"])
    inv.compter(s, "a")
    etat = inv.compter(s, "a")
    assert etat["comptees_plusieurs_fois"] == ["a"]
    assert etat["trouvees"] == 1   # comptée deux fois, mais trouvée une seule


def test_annulation_corrige_sans_rien_effacer(tmp_path):
    """Se tromper doit se corriger : on ajoute une annulation, le fichier reste en ajout seul."""
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a", "b"])
    inv.compter(s, "a")
    etat = inv.annuler_comptage(s, "a")
    assert etat["trouvees"] == 0
    assert etat["manquantes"] == ["a", "b"]
    assert len(inv._lignes(s)) == 3, "l'historique des corrections est conservé"


def test_annulation_ne_descend_pas_sous_zero(tmp_path):
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a"])
    etat = inv.annuler_comptage(s, "a")
    assert etat["trouvees"] == 0


def test_inventaire_clos_refuse_les_comptages(tmp_path):
    inv = Inventaire(tmp_path)
    s = inv.demarrer(["a"])
    assert inv.cloturer(s)["statut"] == "clos"
    with pytest.raises(ValueError, match="clos"):
        inv.compter(s, "a")


def test_session_inconnue_refusee(tmp_path):
    inv = Inventaire(tmp_path)
    for action in (lambda: inv.etat("zzz"), lambda: inv.compter("zzz", "a"),
                   lambda: inv.cloturer("zzz")):
        with pytest.raises(KeyError):
            action()


def test_sessions_listees_de_la_plus_recente(tmp_path):
    inv = Inventaire(tmp_path)
    a = inv.demarrer(["x"], libelle="ancien")
    b = inv.demarrer(["y"], libelle="recent")
    assert {s["session"] for s in inv.sessions()} == {a, b}


def test_aucune_session_ne_plante_pas(tmp_path):
    assert Inventaire(tmp_path / "vide").sessions() == []
