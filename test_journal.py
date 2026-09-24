from journal import Journal


def _resultat(montures, fiable=True):
    return {"resultats": [{"monture": m, "similarite": 0.9} for m in montures], "fiable": fiable}


def test_enregistre_et_relit_une_prediction(tmp_path):
    j = Journal(tmp_path)
    ident = j.enregistrer(type_prediction="similarite", resultat=_resultat(["a", "b"]),
                          version="dinov2_large", photo=b"\xff\xd8fake", latence_ms=120)
    (p,) = j.predictions()
    assert p["id"] == ident
    assert p["monture_choisie"] is None
    assert (tmp_path / "photos" / f"{ident}.jpg").read_bytes() == b"\xff\xd8fake"


def test_le_choix_de_lopticien_complete_la_prediction(tmp_path):
    j = Journal(tmp_path)
    ident = j.enregistrer(type_prediction="similarite", resultat=_resultat(["a", "b"]),
                          version="v1")
    assert j.noter_choix(ident, "b") is True
    (p,) = j.predictions()
    assert p["monture_choisie"] == "b"
    assert p["resultat"]["resultats"], "le résultat d'origine doit être conservé"


def test_choix_sur_prediction_inconnue_refuse(tmp_path):
    j = Journal(tmp_path)
    assert j.noter_choix("inexistant", "a") is False


def test_journal_en_ajout_seul_garde_lhistorique(tmp_path):
    """Une correction est ajoutée, jamais substituée : un fichier en ajout seul ne peut pas être
    corrompu par une écriture concurrente."""
    j = Journal(tmp_path)
    ident = j.enregistrer(type_prediction="similarite", resultat=_resultat(["a"]), version="v1")
    j.noter_choix(ident, "a")
    j.noter_choix(ident, "b")   # l'opticien se corrige
    assert len(j._lire_brut()) == 3
    (p,) = j.predictions()
    assert p["monture_choisie"] == "b", "la dernière correction fait foi"


def test_bilan_calcule_le_recall_reel(tmp_path):
    j = Journal(tmp_path)
    # bonne monture en tête
    i1 = j.enregistrer(type_prediction="similarite", resultat=_resultat(["a", "b", "c"]), version="v")
    j.noter_choix(i1, "a")
    # bonne monture en 3e position : ratée à k=1, trouvée à k=5
    i2 = j.enregistrer(type_prediction="similarite", resultat=_resultat(["x", "y", "b"]), version="v")
    j.noter_choix(i2, "b")
    # pas encore validée par l'opticien -> exclue du calcul
    j.enregistrer(type_prediction="similarite", resultat=_resultat(["z"]), version="v")

    b = j.bilan()
    assert b["predictions"] == 3
    assert b["validees"] == 2
    assert b["recall@1"] == 0.5
    assert b["recall@5"] == 1.0


def test_bilan_sans_validation_ne_pretend_rien(tmp_path):
    j = Journal(tmp_path)
    j.enregistrer(type_prediction="similarite", resultat=_resultat(["a"]), version="v")
    b = j.bilan()
    assert b["validees"] == 0
    assert "recall@1" not in b, "aucun recall ne doit être annoncé sans vérité terrain"
