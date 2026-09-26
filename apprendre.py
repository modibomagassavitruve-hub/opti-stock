"""Ferme la boucle : ce que l'usage réel apprend au modèle.

Jusqu'ici le journal était à sens unique. Les identifications y entraient, `bilan()` en sortait
un chiffre, et rien ne revenait au modèle. Deux choses manquaient, et ce module les fait :

1. RECALIBRER LE SEUIL. Le seuil livré (0.777) a été calibré sur des photos prises en une seule
   rafale -- même lumière, même fond, même journée. La réalité est plus dure. Un seuil trop
   permissif fait afficher « correspondance nette » sur des cas où le modèle a tort, et
   l'opticien commande la mauvaise référence : c'est exactement le risque contre lequel le
   seuil existe, et il n'était pas couvert en conditions réelles.

2. VERSER LES PHOTOS VALIDÉES au jeu d'entraînement. Chaque photo validée est un exemple
   étiqueté pris en conditions réelles -- précisément ce qui manque au catalogue. C'est le seul
   apport qui puisse encore faire progresser le modèle : reprendre des photos du même
   présentoir n'y changerait rien, la courbe d'apprentissage est plate.

Une distinction porte tout le module : « aucune ne correspond » est une donnée, pas un vide.
Elle dit que la première proposition était fausse -- l'information la plus utile pour régler un
seuil. Mais elle ne dit pas quelle était la bonne monture : elle ne peut donc pas servir
d'exemple d'entraînement. Elle compte pour le seuil, pas pour les photos.

Rien n'est appliqué sans demande explicite. Sur une poignée de validations, une recalibration
automatique remplacerait un seuil imparfait par un seuil tiré du bruit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from entrainer_tete import calibrer_seuil
from journal import Journal

# En dessous, l'estimation est trop bruitée pour qu'on y touche. Pour une proportion autour de
# 0.9, trente cas donnent déjà +/- 11 points à 95 % : c'est un plancher, pas un confort.
MINIMUM_VALIDATIONS = 30
SANS_CORRESPONDANCE = ("aucune", "")


def cas_valides(journal: Journal) -> list[dict]:
    """Prédictions tranchées par un humain, avec leur similarité de tête de liste.

    `juste` vaut False quand l'opticien a répondu « aucune » : la première proposition était
    bien fausse. C'est une donnée, et une des plus utiles pour le seuil.
    """
    cas = []
    for p in journal.predictions():
        choisie = p.get("monture_choisie")
        resultats = (p.get("resultat") or {}).get("resultats") or []
        if not choisie or not resultats:
            continue
        cas.append({
            "id": p["id"],
            "similarite": float(resultats[0]["similarite"]),
            "monture_choisie": choisie,
            "juste": resultats[0]["monture"] == choisie,
            "etiquetee": choisie not in SANS_CORRESPONDANCE,
            "fiable_annonce": bool((p.get("resultat") or {}).get("fiable")),
        })
    return cas


def _mesure(cas: list[dict], seuil: float) -> dict:
    """Ce qu'un seuil donnerait sur ces cas : combien de réponses, et combien de justes."""
    if not cas:
        return {"couverture": 0.0, "precision": None, "repondus": 0}
    gardes = [c for c in cas if c["similarite"] >= seuil]
    return {
        "couverture": len(gardes) / len(cas),
        "precision": (sum(c["juste"] for c in gardes) / len(gardes)) if gardes else None,
        "repondus": len(gardes),
    }


def seuil_reel(journal: Journal, tete_json: Path, precision_visee: float = 0.90,
                minimum: int = MINIMUM_VALIDATIONS) -> dict:
    """Compare le seuil en service à celui que l'usage réel réclame."""
    cas = cas_valides(journal)
    actuel = None
    if Path(tete_json).is_file():
        actuel = json.loads(Path(tete_json).read_text(encoding="utf-8")).get("seuil_confiance")

    rapport = {
        "validations": len(cas),
        "minimum": minimum,
        "assez": len(cas) >= minimum,
        "seuil_actuel": actuel,
        "precision_visee": precision_visee,
    }
    if not cas:
        return rapport

    rapport["reel_au_seuil_actuel"] = _mesure(cas, actuel) if actuel is not None else None
    if not rapport["assez"]:
        # On mesure quand même, pour que le chiffre soit visible ; on ne propose rien.
        return rapport

    propose = calibrer_seuil(np.array([c["similarite"] for c in cas]),
                              np.array([c["juste"] for c in cas]), precision_visee)
    rapport["seuil_propose"] = propose
    rapport["reel_au_seuil_propose"] = _mesure(cas, propose)
    # 1.01 est la valeur de refus de calibrer_seuil : aucun seuil n'atteint la précision visée.
    rapport["atteignable"] = propose <= 1.0
    return rapport


def appliquer_seuil(tete_json: Path, seuil: float) -> None:
    """Écrit le nouveau seuil à côté de la tête, en gardant l'ancien pour pouvoir revenir."""
    chemin = Path(tete_json)
    config = json.loads(chemin.read_text(encoding="utf-8")) if chemin.is_file() else {}
    config["seuil_precedent"] = config.get("seuil_confiance")
    config["seuil_confiance"] = seuil
    config["seuil_origine"] = "journal"
    chemin.write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------- photos
def _deja_versees(racine: Path) -> set[str]:
    """Identifiants déjà versés, lus depuis les noms de fichiers eux-mêmes : pas de registre
    séparé à tenir synchronisé, et supprimer une photo suffit à défaire le versement."""
    return {c.stem.removeprefix("journal_")
            for c in Path(racine).glob("*/journal_*.jpg")}


def photos_a_verser(journal: Journal, racine: Path) -> list[dict]:
    """Cas étiquetés dont la photo existe et n'a pas encore été versée.

    Les « aucune » en sont exclus : ils disent que la proposition était fausse, pas quelle
    était la bonne monture. Sans étiquette, pas d'exemple d'entraînement.
    """
    deja = _deja_versees(racine)
    a_verser = []
    for c in cas_valides(journal):
        if not c["etiquetee"] or c["id"] in deja:
            continue
        photo = journal.photos / f"{c['id']}.jpg"
        if photo.is_file():
            a_verser.append({**c, "photo": photo})
    return a_verser


def verser_photos(journal: Journal, racine: Path) -> list[dict]:
    """Copie les photos validées dans le dossier de leur monture, pour que le prochain
    `mettre_a_jour.py` les prenne.

    Le nom `journal_<id>.jpg` rend le versement identifiable et réversible : une photo mal
    étiquetée -- l'opticien a cliqué à côté -- se retire en supprimant le fichier, et le journal
    en garde la trace de toute façon.
    """
    verses = []
    for c in photos_a_verser(journal, racine):
        dossier = Path(racine) / c["monture_choisie"]
        dossier.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c["photo"], dossier / f"journal_{c['id']}.jpg")
        verses.append(c)
    return verses


# ------------------------------------------------------------------------ CLI
def _pourcent(x) -> str:
    return "—" if x is None else f"{x * 100:.0f} %"


def rapport_texte(rapport: dict, a_verser: list[dict]) -> str:
    lignes = [f"Validations exploitables : {rapport['validations']}"]

    reel = rapport.get("reel_au_seuil_actuel")
    if reel and rapport["seuil_actuel"] is not None:
        lignes += [
            "",
            f"Seuil en service : {rapport['seuil_actuel']:.3f}",
            f"  sur l'usage réel : répond {_pourcent(reel['couverture'])} du temps, "
            f"juste {_pourcent(reel['precision'])} ({reel['repondus']} cas)",
        ]

    if not rapport["assez"]:
        lignes += [
            "",
            f"Pas de recalibration proposée : il faut au moins {rapport['minimum']} validations, "
            f"il y en a {rapport['validations']}.",
            "Sur si peu de cas, un seuil recalculé viendrait du bruit et serait pire que celui",
            "en place. Laisser l'usage remplir le journal.",
        ]
    elif not rapport.get("atteignable"):
        lignes += [
            "",
            f"Aucun seuil n'atteint {_pourcent(rapport['precision_visee'])} de justesse sur "
            "l'usage réel.",
            "L'API ne devrait donc jamais se déclarer fiable : c'est le bon comportement,",
            "mieux vaut ne rien affirmer que d'affirmer sur du bruit.",
        ]
    else:
        prop = rapport["reel_au_seuil_propose"]
        lignes += [
            "",
            f"Seuil proposé : {rapport['seuil_propose']:.3f}",
            f"  répondrait {_pourcent(prop['couverture'])} du temps, "
            f"juste {_pourcent(prop['precision'])} ({prop['repondus']} cas)",
            "",
            "Appliquer :  python apprendre.py --appliquer-seuil",
        ]

    lignes += ["", f"Photos validées prêtes à verser à l'entraînement : {len(a_verser)}"]
    if a_verser:
        lignes += ["  python apprendre.py --verser-photos   puis   python mettre_a_jour.py"]
    return "\n".join(lignes)


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parseur.add_argument("--journal", default="data/journal")
    parseur.add_argument("--tete-json", default="data/tete.json")
    parseur.add_argument("--racine", default="data/mes_montures")
    parseur.add_argument("--precision-visee", type=float, default=0.90)
    parseur.add_argument("--minimum", type=int, default=MINIMUM_VALIDATIONS)
    parseur.add_argument("--appliquer-seuil", action="store_true",
                         help="écrit le seuil proposé (refusé si les validations manquent)")
    parseur.add_argument("--verser-photos", action="store_true",
                         help="copie les photos validées dans data/mes_montures/<monture>/")
    args = parseur.parse_args()

    journal = Journal(Path(args.journal))
    rapport = seuil_reel(journal, Path(args.tete_json), args.precision_visee, args.minimum)
    a_verser = photos_a_verser(journal, Path(args.racine))
    print(rapport_texte(rapport, a_verser))

    if args.appliquer_seuil:
        print()
        if not rapport["assez"]:
            raise SystemExit("Refusé : pas assez de validations pour recalibrer.")
        if not rapport.get("atteignable"):
            raise SystemExit("Refusé : aucun seuil n'atteint la précision visée.")
        appliquer_seuil(Path(args.tete_json), rapport["seuil_propose"])
        print(f"Seuil écrit : {rapport['seuil_propose']:.3f} "
              f"(précédent conservé dans {args.tete_json})")
        print("Redémarrer l'API pour qu'elle le charge.")

    if args.verser_photos:
        print()
        verses = verser_photos(journal, Path(args.racine))
        print(f"{len(verses)} photo(s) versée(s) dans {args.racine}/")
        if verses:
            print("Relancer `python mettre_a_jour.py` pour les intégrer au catalogue.")


if __name__ == "__main__":
    main()
