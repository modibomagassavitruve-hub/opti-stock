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
