"""Le réseau manipule des messages privés entre entreprises : les tests portent surtout sur
ce qui ne doit PAS être possible."""
import pytest

from reseau import Reseau, cle_monture


@pytest.fixture
def reseau(tmp_path):
    return Reseau(tmp_path / "reseau.jsonl")


@pytest.fixture
def deux_boutiques(reseau):
    a = reseau.inscrire("Optique Centre", "Lyon", "a@ex.fr")
    b = reseau.inscrire("Vision Plus", "Villeurbanne", "b@ex.fr")
    return reseau, a, b


# ------------------------------------------------------------------ inscription
def test_inscription_donne_un_identifiant_et_un_jeton(reseau):
    b = reseau.inscrire("Optique Centre", "Lyon", "contact@ex.fr")
    assert b["boutique"] and b["jeton"]
    assert reseau.authentifier(b["jeton"]) == b["boutique"]


def test_deux_boutiques_ont_des_jetons_differents(deux_boutiques):
    _, a, b = deux_boutiques
    assert a["jeton"] != b["jeton"]
    assert a["boutique"] != b["boutique"]


def test_jeton_inconnu_n_authentifie_personne(reseau):
    reseau.inscrire("Optique Centre", "Lyon", "a@ex.fr")
    assert reseau.authentifier("pas-un-jeton") is None
    assert reseau.authentifier("") is None


def test_l_annuaire_n_expose_ni_jeton_ni_email(deux_boutiques):
    """Le jeton vaut mot de passe ; l'e-mail est une donnée personnelle. Ni l'un ni l'autre ne
    doit sortir de l'annuaire, qui est lisible par toutes les boutiques inscrites."""
    reseau, _, _ = deux_boutiques
    for fiche in reseau.boutiques().values():
        assert "jeton" not in fiche
        assert "email" not in fiche
        assert fiche["nom"]


# ---------------------------------------------------------------------- partage
def test_chercher_par_reference_trouve_la_bonne_boutique(deux_boutiques):
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866", "libelle": "50"}])

    trouves = reseau.chercher(marque="OCTIKA", reference="OS866",
                               sauf_boutique=a["boutique"])
    assert len(trouves) == 1
    assert trouves[0]["nom"] == "Vision Plus"


def test_reference_tolere_les_confusions_de_l_ocr(deux_boutiques):
    """La référence est gravée sur la branche et relue par OCR : O/0 et I/1 se confondent."""
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])
    assert reseau.chercher(marque="OCTIKA", reference="0S866",
                            sauf_boutique=a["boutique"])


def test_une_boutique_ne_se_trouve_pas_elle_meme(deux_boutiques):
    reseau, a, _ = deux_boutiques
    reseau.partager(a["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])
    assert reseau.chercher(marque="OCTIKA", sauf_boutique=a["boutique"]) == []


def test_seul_le_stock_partage_est_visible(deux_boutiques):
    """Le reste du stock d'une boutique ne doit jamais apparaître."""
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])
    assert reseau.chercher(marque="GIGI STUDIOS", sauf_boutique=a["boutique"]) == []


def test_un_nouveau_partage_remplace_le_precedent(deux_boutiques):
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])
    reseau.partager(b["boutique"], [{"marque": "GIGI STUDIOS", "reference": "6135"}])
    assert reseau.chercher(marque="OCTIKA", sauf_boutique=a["boutique"]) == []
    assert reseau.chercher(marque="GIGI STUDIOS", sauf_boutique=a["boutique"])


def test_partage_ignore_les_montures_sans_marque(deux_boutiques):
    """Sans marque, une monture n'est identifiable par personne d'autre."""
    reseau, a, b = deux_boutiques
    assert reseau.partager(b["boutique"], [{"marque": "", "libelle": "50"},
                                            {"marque": "OCTIKA", "reference": "OS866"}]) == 1


# ----------------------------------------------------------------- conversation
def test_flux_complet_demande_puis_reponse(deux_boutiques):
    """Le parcours de la démonstration : A cherche, trouve chez B, écrit, B répond."""
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])

    trouve = reseau.chercher(marque="OCTIKA", reference="OS866",
                             sauf_boutique=a["boutique"])[0]
    conv = reseau.ouvrir_conversation(a["boutique"], trouve["boutique"], "OCTIKA OS866")
    reseau.ecrire(conv, a["boutique"], "Bonjour, cette monture est-elle encore disponible ?")
    reseau.ecrire(conv, b["boutique"], "Oui, je vous la mets de côté.")

    fil = reseau.messages(conv, a["boutique"])
    assert [m["auteur"] for m in fil] == [a["boutique"], b["boutique"]]
    assert "disponible" in fil[0]["texte"]


def test_une_seule_conversation_par_paire_et_monture(deux_boutiques):
    reseau, a, b = deux_boutiques
    c1 = reseau.ouvrir_conversation(a["boutique"], b["boutique"], "OCTIKA OS866")
    c2 = reseau.ouvrir_conversation(a["boutique"], b["boutique"], "OCTIKA OS866")
    c3 = reseau.ouvrir_conversation(a["boutique"], b["boutique"], "GIGI 6135")
    assert c1 == c2 != c3


def test_une_boutique_ne_se_contacte_pas_elle_meme(deux_boutiques):
    reseau, a, _ = deux_boutiques
    with pytest.raises(ValueError):
        reseau.ouvrir_conversation(a["boutique"], a["boutique"])


def test_un_tiers_ne_peut_pas_lire_la_conversation(reseau):
    """Le point le plus important du module : l'étanchéité entre boutiques."""
    a = reseau.inscrire("A", "Lyon", "a@ex.fr")
    b = reseau.inscrire("B", "Lyon", "b@ex.fr")
    c = reseau.inscrire("C", "Lyon", "c@ex.fr")

    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    reseau.ecrire(conv, a["boutique"], "Prix confidentiel : 40 EUR")

    with pytest.raises(PermissionError):
        reseau.messages(conv, c["boutique"])
    with pytest.raises(PermissionError):
        reseau.ecrire(conv, c["boutique"], "je m'incruste")


def test_conversation_inconnue_est_une_erreur(deux_boutiques):
    reseau, a, _ = deux_boutiques
    with pytest.raises(KeyError):
        reseau.messages("inexistante", a["boutique"])
    with pytest.raises(KeyError):
        reseau.ecrire("inexistante", a["boutique"], "bonjour")


def test_message_vide_refuse(deux_boutiques):
    reseau, a, b = deux_boutiques
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    with pytest.raises(ValueError):
        reseau.ecrire(conv, a["boutique"], "   ")


def test_message_tres_long_tronque(deux_boutiques):
    reseau, a, b = deux_boutiques
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    m = reseau.ecrire(conv, a["boutique"], "x" * 5000)
    assert len(m["texte"]) == 2000


def test_liste_des_conversations_des_deux_cotes(deux_boutiques):
    reseau, a, b = deux_boutiques
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"], "OCTIKA OS866")
    reseau.ecrire(conv, a["boutique"], "Disponible ?")

    cote_a = reseau.conversations(a["boutique"])
    cote_b = reseau.conversations(b["boutique"])
    assert len(cote_a) == len(cote_b) == 1
    assert cote_a[0]["interlocuteur"] == "Vision Plus"
    assert cote_b[0]["interlocuteur"] == "Optique Centre"
    assert cote_b[0]["dernier"] == "Disponible ?"


def test_conversations_d_une_boutique_sans_echange(deux_boutiques):
    reseau, a, _ = deux_boutiques
    assert reseau.conversations(a["boutique"]) == []


# -------------------------------------------------------------------- blocage
def test_boutique_bloquee_ne_peut_plus_ouvrir_de_conversation(deux_boutiques):
    reseau, a, b = deux_boutiques
    reseau.bloquer(b["boutique"], a["boutique"])
    with pytest.raises(PermissionError):
        reseau.ouvrir_conversation(a["boutique"], b["boutique"])


def test_blocage_coupe_une_conversation_en_cours(deux_boutiques):
    reseau, a, b = deux_boutiques
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    reseau.ecrire(conv, a["boutique"], "Bonjour")
    reseau.bloquer(b["boutique"], a["boutique"])

    with pytest.raises(PermissionError):
        reseau.ecrire(conv, a["boutique"], "Rebonjour")
    reseau.ecrire(conv, b["boutique"], "Je préfère en rester là")  # B garde la parole


def test_boutique_bloquee_ne_voit_plus_le_stock_partage(deux_boutiques):
    reseau, a, b = deux_boutiques
    reseau.partager(b["boutique"], [{"marque": "OCTIKA", "reference": "OS866"}])
    reseau.bloquer(b["boutique"], a["boutique"])
    assert reseau.chercher(marque="OCTIKA", sauf_boutique=a["boutique"]) == []


def test_deblocage_retablit_le_contact(deux_boutiques):
    reseau, a, b = deux_boutiques
    reseau.bloquer(b["boutique"], a["boutique"])
    reseau.debloquer(b["boutique"], a["boutique"])
    assert reseau.ouvrir_conversation(a["boutique"], b["boutique"])


def test_blocage_n_est_pas_reciproque(deux_boutiques):
    """A bloqué par B : B peut toujours contacter A."""
    reseau, a, b = deux_boutiques
    reseau.bloquer(b["boutique"], a["boutique"])
    assert reseau.ouvrir_conversation(b["boutique"], a["boutique"])


def test_une_boutique_ne_se_bloque_pas_elle_meme(deux_boutiques):
    reseau, a, _ = deux_boutiques
    with pytest.raises(ValueError):
        reseau.bloquer(a["boutique"], a["boutique"])


# ---------------------------------------------------------------- signalement
def test_signalement_conserve(deux_boutiques):
    reseau, a, b = deux_boutiques
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    reseau.signaler(conv, a["boutique"], "Propos insultants")
    assert reseau.signalements()[0]["motif"] == "Propos insultants"


# ------------------------------------------------------------------ stockage
def test_les_ecritures_sont_en_ajout_seul(reseau):
    """Comme le journal : rien n'est jamais réécrit, l'historique reste vérifiable."""
    a = reseau.inscrire("A", "Lyon", "a@ex.fr")
    b = reseau.inscrire("B", "Lyon", "b@ex.fr")
    conv = reseau.ouvrir_conversation(a["boutique"], b["boutique"])
    reseau.ecrire(conv, a["boutique"], "un")
    avant = reseau.fichier.read_text()

    reseau.ecrire(conv, a["boutique"], "deux")
    assert reseau.fichier.read_text().startswith(avant)


def test_reseau_vide_ne_plante_pas(reseau):
    assert reseau.boutiques() == {}
    assert reseau.chercher(marque="OCTIKA") == []
    assert reseau.signalements() == []


def test_cle_monture_normalise():
    assert cle_monture("OCTIKA", "OS 866") == cle_monture("octika", "0s-866")
    assert cle_monture("OCTIKA", "OS866") != cle_monture("GIGI", "OS866")
