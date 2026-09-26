"""Fiches montures saisies en boutique : marque, référence, photo (module 1).

Pourquoi ce module existe alors qu'il y a déjà un catalogue. Le catalogue (`catalogue.npz`)
est le jeu de montures sur lequel le modèle a été entraîné : on ne peut pas y ajouter une
monture sans tout reconstruire, ce qui prend des minutes et un ordinateur. Or l'opticien qui
reçoit une monture inconnue doit pouvoir l'entrer en stock tout de suite, depuis son
téléphone, sans que personne ne réentraîne quoi que ce soit.

Une fiche est donc l'identité d'une monture -- marque, référence, coloris, calibre, photo --
indépendante du modèle. Le stock compte des quantités par identifiant ; ces identifiants
viennent soit du catalogue entraîné, soit d'ici. Les deux cohabitent sans se gêner.

Cette séparation est celle du schéma : la table `monture` porte l'identité, `stock_item` porte
la quantité. On ne fait que la respecter.

Les photos saisies alimenteront le modèle plus tard, par `apprendre.py` et un réentraînement.
Elles sont donc conservées telles quelles, en pleine résolution.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

CHAMPS = ("marque", "reference", "coloris", "calibre", "pont", "branche")


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Fiches:
    def __init__(self, dossier: Path, photos: Path | None = None):
        self.dossier = Path(dossier)
        self.lignes = self.dossier / "fiches.jsonl"
        # Les photos peuvent être rangées hors du dossier de la boutique. C'est ce que fait
        # l'API : une balise <img> ne peut pas porter d'en-tête d'authentification, donc la
        # photo est adressée par son seul identifiant, tiré au hasard et non énumérable.
        # Les fiches elles-mêmes -- marque, référence, stock -- restent cloisonnées.
        self.photos = Path(photos) if photos else self.dossier / "photos_fiches"

    def _lire(self) -> list[dict]:
        if not self.lignes.exists():
            return []
        with open(self.lignes, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def creer(self, champs: dict, photo: bytes | None = None) -> dict:
        """Crée une fiche. Seule la marque est exigée : au comptoir, la référence n'est pas
        toujours lisible, et refuser la saisie pour ça ferait abandonner l'opticien.

        L'identifiant est préfixé `f_` pour qu'on distingue toujours, dans le stock et dans
        l'inventaire, ce qui vient d'une saisie de ce qui vient du catalogue entraîné.
        """
        marque = (champs.get("marque") or "").strip()
        if not marque:
            raise ValueError("La marque est nécessaire : sans elle la monture est introuvable")

        identifiant = "f_" + uuid.uuid4().hex[:8]
        self.dossier.mkdir(parents=True, exist_ok=True)
        if photo:
            self.photos.mkdir(parents=True, exist_ok=True)
            (self.photos / f"{identifiant}.jpg").write_bytes(photo)

        fiche = {"monture": identifiant, "date": _horodatage(),
                 "photo": bool(photo),
                 **{c: (champs.get(c) or "").strip() for c in CHAMPS}}
        with open(self.lignes, "a", encoding="utf-8") as f:
            f.write(json.dumps(fiche, ensure_ascii=False) + "\n")
        return fiche

    def modifier(self, monture: str, champs: dict) -> dict:
        """Corrige une fiche. Comme partout ici, on ajoute une ligne plutôt que d'en réécrire
        une : l'historique des corrections reste lisible."""
        if monture not in self.toutes():
            raise KeyError(f"Fiche {monture} inconnue")
        correction = {"monture": monture, "date": _horodatage(),
                      **{c: champs[c].strip() for c in CHAMPS if c in champs}}
        with open(self.lignes, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction, ensure_ascii=False) + "\n")
        return self.toutes()[monture]

    def toutes(self) -> dict[str, dict]:
        """Fiches consolidées : les corrections successives s'appliquent sur l'originale."""
        par_id: dict[str, dict] = {}
        for ligne in self._lire():
            identifiant = ligne["monture"]
            if identifiant in par_id:
                par_id[identifiant].update(ligne)
            else:
                par_id[identifiant] = dict(ligne)
        return par_id

    def fiche(self, monture: str) -> dict | None:
        return self.toutes().get(monture)

    def photo(self, monture: str) -> Path | None:
        chemin = self.photos / f"{monture}.jpg"
        return chemin if chemin.is_file() else None

    def libelle(self, monture: str) -> str:
        """Ce qu'on affiche : « OCTIKA OS866 », ou la marque seule si la référence manque."""
        f = self.fiche(monture)
        if not f:
            return monture
        return " ".join(x for x in (f.get("marque"), f.get("reference")) if x) or monture

    def chercher(self, texte: str) -> list[dict]:
        """Recherche libre sur marque et référence, pour retrouver une fiche au comptoir."""
        besoin = texte.strip().casefold()
        if not besoin:
            return sorted(self.toutes().values(), key=lambda f: (f.get("marque", ""),
                                                                  f.get("reference", "")))
        return [f for f in self.toutes().values()
                if besoin in f"{f.get('marque', '')} {f.get('reference', '')}".casefold()]
