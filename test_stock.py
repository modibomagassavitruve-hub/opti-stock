"""Le stock porte des chiffres sur lesquels un opticien commande et vend : les tests portent
surtout sur ce qui doit rester impossible, et sur la traçabilité des écarts."""
import pytest

from stock import Stock


@pytest.fixture
def stock(tmp_path):
    return Stock(tmp_path / "stock.jsonl")


# ------------------------------------------------------------------- entrées
def test_entree_puis_quantite(stock):
    stock.entrer("18", 3, "vitrine A")
    assert stock.quantite("18") == 3
    assert stock.etat_monture("18")["emplacement"] == "vitrine A"


def test_entrees_successives_s_additionnent(stock):
    stock.entrer("18", 2)
    stock.entrer("18", 3)
    assert stock.quantite("18") == 5


def test_monture_jamais_vue_est_a_zero(stock):
    assert stock.quantite("inconnue") == 0
    assert stock.etat_monture("inconnue")["quantite"] == 0


def test_entree_de_zero_refusee(stock):
    with pytest.raises(ValueError):
        stock.entrer("18", 0)


def test_entree_negative_refusee(stock):
    with pytest.raises(ValueError):
        stock.entrer("18", -5)


def test_motif_d_entree_inconnu_refuse(stock):
    with pytest.raises(ValueError, match="Motif"):
        stock.entrer("18", 1, motif="cadeau_de_noel")


# ------------------------------------------------------------------- sorties
def test_vente_diminue_le_stock(stock):
    stock.entrer("18", 3)
    stock.sortir("18", 1, motif="vente")
    assert stock.quantite("18") == 2


def test_vendre_plus_que_le_stock_est_refuse(stock):
    """Un stock négatif n'existe pas en rayon : l'accepter rendrait tous les chiffres douteux."""
    stock.entrer("18", 2)
    with pytest.raises(ValueError, match="refusée"):
        stock.sortir("18", 3)
    assert stock.quantite("18") == 2, "un refus ne doit rien modifier"


def test_vendre_une_monture_absente_est_refuse(stock):
    with pytest.raises(ValueError):
        stock.sortir("jamais_vue", 1)


def test_vendre_exactement_le_stock_restant(stock):
    stock.entrer("18", 2)
    stock.sortir("18", 2)
    assert stock.quantite("18") == 0


def test_motif_de_sortie_inconnu_refuse(stock):
    stock.entrer("18", 1)
    with pytest.raises(ValueError, match="Motif"):
        stock.sortir("18", 1, motif="evaporation")


def test_casse_et_perte_sont_des_sorties_legitimes(stock):
    stock.entrer("18", 3)
    stock.sortir("18", 1, motif="casse")
    stock.sortir("18", 1, motif="perte")
    assert stock.quantite("18") == 1


# --------------------------------------------------------------- ajustements
def test_ajustement_pose_la_quantite_constatee(stock):
    stock.entrer("18", 5)
    stock.ajuster("18", 3)
    assert stock.quantite("18") == 3


def test_ajustement_enregistre_l_ecart(stock):
    """C'est ce chiffre que l'inventaire doit pouvoir montrer : ce qui manque, et de combien."""
    stock.entrer("18", 5)
    stock.ajuster("18", 3)
    assert [m for m in stock.mouvements("18") if m["type"] == "ajustement"][0]["ecart"] == -2


def test_ajustement_peut_augmenter(stock):
    stock.entrer("18", 2)
    stock.ajuster("18", 4)
    assert stock.quantite("18") == 4


def test_ajustement_negatif_refuse(stock):
    with pytest.raises(ValueError):
        stock.ajuster("18", -1)


def test_ajustement_puis_vente_repart_du_constate(stock):
    stock.entrer("18", 5)
    stock.ajuster("18", 2)
    stock.sortir("18", 2)
    assert stock.quantite("18") == 0


def test_vendre_plus_que_le_constate_apres_ajustement_refuse(stock):
    stock.entrer("18", 5)
    stock.ajuster("18", 1)
    with pytest.raises(ValueError):
        stock.sortir("18", 2)


# -------------------------------------------------------------- emplacements
def test_deplacer_change_l_emplacement_sans_toucher_la_quantite(stock):
    stock.entrer("18", 2, "vitrine A")
    stock.deplacer("18", "réserve")
    e = stock.etat_monture("18")
    assert (e["emplacement"], e["quantite"]) == ("réserve", 2)


def test_derniere_declaration_d_emplacement_gagne(stock):
    stock.entrer("18", 1, "vitrine A")
    stock.entrer("18", 1, "vitrine B")
    assert stock.etat_monture("18")["emplacement"] == "vitrine B"


def test_entree_sans_emplacement_ne_l_efface_pas(stock):
    stock.entrer("18", 1, "vitrine A")
    stock.entrer("18", 1)
    assert stock.etat_monture("18")["emplacement"] == "vitrine A"


# --------------------------------------------------------------------- états
def test_en_stock_exclut_les_epuisees(stock):
    """Ce que le réseau a le droit d'exposer : sinon un confrère se déplace pour rien."""
    stock.entrer("18", 1)
    stock.entrer("31", 1)
    stock.sortir("18", 1)
    assert set(stock.en_stock()) == {"31"}


def test_une_monture_epuisee_reste_connue(stock):
    """« J'en avais, je n'en ai plus » n'est pas « je n'en ai jamais eu »."""
    stock.entrer("18", 1)
    stock.sortir("18", 1)
    assert "18" in stock.etat()
    assert stock.etat()["18"]["quantite"] == 0


def test_bilan(stock):
    stock.entrer("18", 3)
    stock.entrer("31", 2)
    stock.sortir("18", 3)
    b = stock.bilan()
    assert (b["references"], b["references_en_stock"], b["pieces"]) == (2, 1, 2)
    assert b["epuisees"] == ["18"]


def test_stock_vide_ne_plante_pas(stock):
    assert stock.etat() == {}
    assert stock.en_stock() == {}
    assert stock.bilan()["pieces"] == 0


# ---------------------------------------------------------------- historique
def test_les_mouvements_sont_conserves_dans_l_ordre(stock):
    stock.entrer("18", 5)
    stock.sortir("18", 2, motif="vente")
    stock.ajuster("18", 1)
    assert [m["type"] for m in stock.mouvements("18")] == ["entree", "sortie", "ajustement"]


def test_mouvements_filtres_par_monture(stock):
    stock.entrer("18", 1)
    stock.entrer("31", 1)
    assert {m["monture"] for m in stock.mouvements("31")} == {"31"}
    assert len(stock.mouvements()) == 2


def test_l_ecriture_est_en_ajout_seul(stock):
    """Un stock qu'on réécrit perd la raison de ses écarts ; c'est justement ce qu'on veut
    pouvoir expliquer."""
    stock.entrer("18", 1)
    avant = stock.fichier.read_text()
    stock.sortir("18", 1)
    assert stock.fichier.read_text().startswith(avant)
