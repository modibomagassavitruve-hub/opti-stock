"""L'inventaire sert à chiffrer un écart : les tests portent sur l'écart, pas sur le comptage."""
import pytest

from inventaire import Inventaire
from stock import Stock


@pytest.fixture
def inv(tmp_path):
    return Inventaire(tmp_path / "inventaires")


@pytest.fixture
def stock(tmp_path):
    return Stock(tmp_path / "stock.jsonl")


def _ecart(etat, monture):
    return [l for l in etat["lignes"] if l["monture"] == monture][0]["ecart"]


# ------------------------------------------------------------------ ouverture
def test_session_fige_le_stock_theorique(inv):
    """Le stock peut bouger pendant l'inventaire -- une vente, une réception -- sans que la
    comparaison finale s'en trouve faussée."""
    s = inv.demarrer({"a": 2, "b": 1})
    etat = inv.etat(s)
    assert etat["pieces_attendues"] == 3
    assert etat["references_attendues"] == 2
    assert etat["statut"] == "en_cours"


def test_libelle_conserve(inv):
    assert inv.etat(inv.demarrer({"a": 1}, "2026"))["libelle"] == "2026"


# --------------------------------------------------------------------- écarts
def test_ce_qui_manque_est_chiffre(inv):
    """« Il m'en manque 2 sur 5 » -- la version qui ne disait que « vue / pas vue » ne pouvait
    pas produire cette phrase."""
    s = inv.demarrer({"a": 5})
    inv.compter(s, "a", 3)
    etat = inv.etat(s)
    assert _ecart(etat, "a") == -2
    assert [l["monture"] for l in etat["manquantes"]] == ["a"]
    assert etat["manquantes"][0]["theorique"] == 5
    assert etat["manquantes"][0]["comptee"] == 3


def test_comptage_exact_ne_laisse_aucun_ecart(inv):
    s = inv.demarrer({"a": 2})
    inv.compter(s, "a", 2)
    etat = inv.etat(s)
    assert etat["manquantes"] == [] and etat["en_trop"] == []


def test_monture_non_comptee_manque_en_entier(inv):
    s = inv.demarrer({"a": 3})
    assert _ecart(inv.etat(s), "a") == -3


def test_surplus_signale(inv):
    """Un retour client non saisi, ou une erreur de rangement : le surplus est une information,
    pas une anomalie à masquer."""
    s = inv.demarrer({"a": 1})
    inv.compter(s, "a", 3)
    etat = inv.etat(s)
    assert _ecart(etat, "a") == 2
    assert [l["monture"] for l in etat["en_trop"]] == ["a"]


def test_monture_trouvee_hors_stock_theorique(inv):
    s = inv.demarrer({"a": 1})
    inv.compter(s, "surprise")
    etat = inv.etat(s)
    assert etat["hors_stock"] == ["surprise"]
    assert _ecart(etat, "surprise") == 1


def test_comptages_successifs_s_additionnent(inv):
    s = inv.demarrer({"a": 5})
    inv.compter(s, "a")
    inv.compter(s, "a", 2)
    assert inv.etat(s)["pieces_comptees"] == 3


# ---------------------------------------------------------------- corrections
def test_annulation_corrige_sans_rien_effacer(inv):
    s = inv.demarrer({"a": 2})
    inv.compter(s, "a", 2)
    inv.annuler_comptage(s, "a")
    assert inv.etat(s)["pieces_comptees"] == 1
    assert len(inv._lignes(s)) == 3, "ouverture, comptage et annulation restent au fichier"


def test_annulation_ne_descend_pas_sous_zero(inv):
    s = inv.demarrer({"a": 1})
    inv.annuler_comptage(s, "a")
    assert inv.etat(s)["pieces_comptees"] == 0


def test_comptage_de_zero_refuse(inv):
    s = inv.demarrer({"a": 1})
    with pytest.raises(ValueError):
        inv.compter(s, "a", 0)


# ------------------------------------------------------------------- clôture
def test_cloturer_applique_les_comptages_au_stock(inv, stock):
    """Le sens du récolement : le rayon fait foi."""
    stock.entrer("a", 5)
    stock.entrer("b", 2)
    s = inv.demarrer({"a": 5, "b": 2}, "2026")
    inv.compter(s, "a", 3)
    inv.compter(s, "b", 2)
    inv.cloturer(s, stock)

    assert stock.quantite("a") == 3, "corrigée d'après le comptage"
    assert stock.quantite("b") == 2, "inchangée, elle était juste"


def test_cloture_laisse_une_trace_de_la_correction(inv, stock):
    stock.entrer("a", 5)
    s = inv.demarrer({"a": 5}, "2026")
    inv.compter(s, "a", 3)
    inv.cloturer(s, stock)
    ajustement = [m for m in stock.mouvements("a") if m["type"] == "ajustement"][0]
    assert ajustement["ecart"] == -2
    assert "2026" in ajustement["motif"]


def test_monture_non_comptee_passe_a_zero_a_la_cloture(inv, stock):
    """Ne pas y toucher laisserait le stock affirmer qu'elle est là, alors qu'on vient de
    constater le contraire. On compte une autre monture pour rester dans un inventaire
    réel -- un inventaire où l'on n'a rien compté du tout est refusé, voir plus bas."""
    stock.entrer("disparue", 2)
    stock.entrer("presente", 1)
    s = inv.demarrer({"disparue": 2, "presente": 1})
    inv.compter(s, "presente", 1)
    inv.cloturer(s, stock)
    assert stock.quantite("disparue") == 0
    assert stock.quantite("presente") == 1


def test_cloture_sans_stock_ne_corrige_rien(inv):
    s = inv.demarrer({"a": 2})
    assert inv.cloturer(s)["statut"] == "clos"


def test_inventaire_clos_refuse_les_comptages(inv):
    s = inv.demarrer({"a": 1})
    inv.cloturer(s)
    with pytest.raises(ValueError, match="clos"):
        inv.compter(s, "a")


def test_double_cloture_refusee(inv, stock):
    """Sans ce garde-fou, reclôturer réappliquerait les ajustements au stock."""
    stock.entrer("a", 5)
    s = inv.demarrer({"a": 5})
    inv.compter(s, "a", 3)
    inv.cloturer(s, stock)
    with pytest.raises(ValueError, match="clos"):
        inv.cloturer(s, stock)


# ------------------------------------------------------------------- sessions
def test_session_inconnue_refusee(inv):
    for appel in (lambda: inv.etat("nope"), lambda: inv.compter("nope", "a"),
                   lambda: inv.cloturer("nope"), lambda: inv.annuler_comptage("nope", "a")):
        with pytest.raises(KeyError):
            appel()


def test_sessions_listees(inv):
    a, b = inv.demarrer({"x": 1}, "ancien"), inv.demarrer({"y": 1}, "recent")
    assert {s["session"] for s in inv.sessions()} == {a, b}


def test_aucune_session_ne_plante_pas(inv):
    assert inv.sessions() == []


def test_cloture_sans_rien_compter_refusee(inv, stock):
    """Clôturer sans avoir compté viderait le stock entier d'un geste. C'est toujours une
    fausse manœuvre -- personne n'ouvre un inventaire pour déclarer sa boutique vide -- et
    c'est arrivé en trois secondes lors d'un test."""
    stock.entrer("a", 5)
    stock.entrer("b", 3)
    s = inv.demarrer({"a": 5, "b": 3})

    with pytest.raises(ValueError, match="Aucune monture comptée"):
        inv.cloturer(s, stock)
    assert stock.quantite("a") == 5, "le stock est intact"
    assert inv.etat(s)["statut"] == "en_cours", "la session reste ouverte"


def test_cloture_a_vide_possible_en_forcant(inv, stock):
    """Le cas réel d'une boutique liquidée doit rester faisable."""
    stock.entrer("a", 5)
    s = inv.demarrer({"a": 5})
    inv.cloturer(s, stock, forcer=True)
    assert stock.quantite("a") == 0


def test_une_seule_monture_comptee_suffit_a_cloturer(inv, stock):
    """Le garde-fou ne vise que le zéro absolu : un inventaire peut légitimement constater
    d'énormes écarts."""
    stock.entrer("a", 5)
    stock.entrer("b", 3)
    s = inv.demarrer({"a": 5, "b": 3})
    inv.compter(s, "a", 1)
    inv.cloturer(s, stock)
    assert (stock.quantite("a"), stock.quantite("b")) == (1, 0)


def test_cloture_sans_stock_reste_libre(inv):
    """Sans stock à corriger, il n'y a rien à protéger."""
    s = inv.demarrer({"a": 5})
    assert inv.cloturer(s)["statut"] == "clos"
