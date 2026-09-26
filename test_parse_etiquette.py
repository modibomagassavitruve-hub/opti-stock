from parse_etiquette import parse_etiquette

MARQUES = ["Ray-Ban", "Persol", "Tom Ford", "Oakley", "Vogue"]


def champs(e):
    return (e.marque, e.reference, e.coloris_code, e.calibre, e.pont, e.branche)


def test_ray_ban_carre_lu_correctement():
    e = parse_etiquette("RAY-BAN RB3025 001/51 58□14 135", MARQUES)
    assert champs(e) == ("Ray-Ban", "RB3025", "001/51", "58", "14", "135")


def test_tirets_et_reference_avec_suffixe():
    e = parse_etiquette("Persol PO3007V 95 52-20 145", MARQUES)
    assert champs(e) == ("Persol", "PO3007V", "95", "52", "20", "145")


def test_reference_en_deux_morceaux_et_marque_en_deux_mots():
    e = parse_etiquette("Ray Ban RB 3025 001/51 58 14 135", MARQUES)
    assert champs(e) == ("Ray-Ban", "RB3025", "001/51", "58", "14", "135")


def test_ocr_bruite():
    e = parse_etiquette("RAYBAN RB3025 OO1/51 58[]14 135", MARQUES)
    assert champs(e) == ("Ray-Ban", "RB3025", "001/51", "58", "14", "135")


def test_carre_lu_comme_zero():
    e = parse_etiquette("Oakley OO9102 26 52018 140", MARQUES)
    assert (e.calibre, e.pont, e.branche) == ("52", "18", "140")


def test_marque_deux_mots_et_tirets_partout():
    e = parse_etiquette("TOM FORD FT5123 001 52-18-140", MARQUES)
    assert champs(e) == ("Tom Ford", "FT5123", "001", "52", "18", "140")


def test_taille_sans_separateur_et_coloris_lettre_chiffre():
    e = parse_etiquette("Vogue VO5051 W44 5218 140", MARQUES)
    assert champs(e) == ("Vogue", "VO5051", "W44", "52", "18", "140")


def test_texte_inexploitable():
    e = parse_etiquette("hello world", MARQUES)
    assert champs(e) == ("", "", "", "", "", "")


def test_deux_tokens_a_trois_chiffres_ne_font_pas_une_reference():
    """Lu sur une vraie étiquette au salon : « OZ EYEV 134 » donnait la référence « EYEV134 »,
    alors que 134 est une longueur de branche ou un prix. Un champ vide se corrige d'un coup
    d'œil ; une valeur fausse se recopie sans la voir."""
    e = parse_etiquette("7706 9 @Z EYEW Pce MURAT C2 OZ EYEV 134", ["OZ EYEWEAR"])
    assert e.reference != "EYEV134"


def test_reference_en_deux_morceaux_toujours_reconnue():
    """La correction ne doit pas casser le cas qu'elle sert : « RB 3025 »."""
    assert parse_etiquette("RAY-BAN RB 3025 58-14 135", ["RAY-BAN"]).reference == "RB3025"


def test_taille_francaise_collee_lue():
    """« 54017-140 » : calibre 54, pont 17, branche 140 -- le carré est lu comme un zéro."""
    e = parse_etiquette("C2 MURAT 54017-140", [])
    assert (e.calibre, e.pont, e.branche) == ("54", "17", "140")
