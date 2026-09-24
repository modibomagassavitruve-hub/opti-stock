"""Met à jour l'appli après un changement de stock, en une commande.

Reconstruit le catalogue puis réentraîne la tête, dans cet ordre. Les deux sont indissociables :
une tête entraînée sur un catalogue périmé ignore les montures ajoutées et son seuil de
confiance est calibré sur un stock qui n'existe plus. Rien n'échoue dans ce cas -- les
dimensions restent compatibles -- donc l'oubli passerait inaperçu. Cette commande le rend
impossible.

Usage :
    # 1. déposer les photos de la nouvelle monture dans data/mes_montures/<identifiant>/
    # 2. renseigner sa marque dans data/montures.csv (améliore nettement la recherche)
    python mettre_a_jour.py
    # 3. redémarrer l'API

    python mettre_a_jour.py --mesurer    # ajoute une évaluation complète (plus long)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def lancer(etape: str, commande: list[str]) -> None:
    print(f"\n{'=' * 70}\n{etape}\n{'=' * 70}", flush=True)
    resultat = subprocess.run([sys.executable] + commande)
    if resultat.returncode != 0:
        raise SystemExit(f"\nÉchec à l'étape « {etape} ». Rien n'a été mis en service : "
                         f"l'API continue de servir le catalogue et la tête précédents.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--racine", type=Path, default=Path("data/mes_montures"))
    p.add_argument("--mesurer", action="store_true",
                    help="évalue la performance après mise à jour (quelques minutes de plus)")
    args = p.parse_args()

    if not args.racine.is_dir():
        raise SystemExit(f"{args.racine} introuvable. Y déposer un dossier par monture.")
    montures = [d for d in args.racine.iterdir() if d.is_dir()]
    photos = sum(len(list(d.glob("*.jpg"))) for d in montures)
    print(f"Stock : {len(montures)} montures, {photos} photos")

    lancer("1/2 — Catalogue", ["construire_catalogue.py", "--racine", str(args.racine)])
    lancer("2/2 — Tête de projection", ["entrainer_tete.py"])

    if args.mesurer:
        lancer("Mesure", ["evaluer.py", "--seuils"])

    print(f"\n{'=' * 70}")
    print("À jour. Redémarrer l'API pour qu'elle recharge le catalogue et la tête :")
    print("    uvicorn app:app --port 8000")


if __name__ == "__main__":
    main()
