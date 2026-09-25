"""Inventaire annuel : pointer le stock réel contre le catalogue (module 2 du schéma).

Différence de nature avec /identifier, qui commande la conception : ici la monture est
forcément au catalogue, et l'opticien la tient en main. Ce n'est pas une identification mais
une CONFIRMATION -- la reconnaissance visuelle propose, l'humain valide d'un geste. Une
reconnaissance imparfaite reste donc utilisable, là où elle serait gênante pour une commande
fournisseur.

Ce que l'inventaire produit de précieux n'est pas la liste de ce qu'on a trouvé, mais celle de
ce qu'on n'a PAS trouvé : montures disparues, vendues sans saisie, ou rangées ailleurs.

Stockage : un JSONL par session, en ajout seul, comme le journal des prédictions. Chaque
comptage est une ligne ; l'état se reconstruit en les relisant. Aucune écriture concurrente ne
peut corrompre le fichier, et se tromper se corrige en recomptant.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Inventaire:
    def __init__(self, dossier: Path):
        self.dossier = Path(dossier)

    def _fichier(self, session: str) -> Path:
        return self.dossier / f"{session}.jsonl"

    def _lignes(self, session: str) -> list[dict]:
        fichier = self._fichier(session)
        if not fichier.exists():
            return []
        with open(fichier, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _ajouter(self, session: str, ligne: dict) -> None:
        self.dossier.mkdir(parents=True, exist_ok=True)
        with open(self._fichier(session), "a", encoding="utf-8") as f:
            f.write(json.dumps({**ligne, "date": _horodatage()}, ensure_ascii=False) + "\n")

    def demarrer(self, montures_attendues: list[str], libelle: str = "") -> str:
        """Ouvre une session en figeant la liste attendue : le catalogue peut changer pendant
        l'inventaire sans fausser le décompte."""
        session = uuid.uuid4().hex[:8]
        self._ajouter(session, {"type": "ouverture", "libelle": libelle,
                                 "attendues": sorted(set(montures_attendues))})
        return session

    def compter(self, session: str, monture: str) -> dict:
        lignes = self._lignes(session)
        if not lignes:
            raise KeyError(f"Inventaire {session} inconnu")
        if self._statut(lignes) == "clos":
            raise ValueError(f"Inventaire {session} déjà clos")
        self._ajouter(session, {"type": "comptage", "monture": monture})
        return self.etat(session)

    def annuler_comptage(self, session: str, monture: str) -> dict:
        """Retire un comptage. On ajoute une ligne d'annulation plutôt que d'en effacer une :
        le fichier reste en ajout seul, et l'historique des corrections reste lisible."""
        if not self._lignes(session):
            raise KeyError(f"Inventaire {session} inconnu")
        self._ajouter(session, {"type": "annulation", "monture": monture})
        return self.etat(session)

    def cloturer(self, session: str) -> dict:
        if not self._lignes(session):
            raise KeyError(f"Inventaire {session} inconnu")
        self._ajouter(session, {"type": "cloture"})
        return self.etat(session)

    @staticmethod
    def _statut(lignes: list[dict]) -> str:
        return "clos" if any(l["type"] == "cloture" for l in lignes) else "en_cours"

    def etat(self, session: str) -> dict:
        lignes = self._lignes(session)
        if not lignes:
            raise KeyError(f"Inventaire {session} inconnu")

        attendues = set(lignes[0].get("attendues", []))
        comptes: dict[str, int] = {}
        for l in lignes:
            if l["type"] == "comptage":
                comptes[l["monture"]] = comptes.get(l["monture"], 0) + 1
            elif l["type"] == "annulation":
                comptes[l["monture"]] = max(0, comptes.get(l["monture"], 0) - 1)
        comptes = {m: n for m, n in comptes.items() if n > 0}

        trouvees = set(comptes)
        return {
            "session": session,
            "libelle": lignes[0].get("libelle", ""),
            "statut": self._statut(lignes),
            "demarre_le": lignes[0]["date"],
            "attendues": len(attendues),
            "trouvees": len(trouvees & attendues),
            "manquantes": sorted(attendues - trouvees),
            "hors_catalogue": sorted(trouvees - attendues),
            "comptees_plusieurs_fois": sorted(m for m, n in comptes.items() if n > 1),
        }

    def sessions(self) -> list[dict]:
        if not self.dossier.is_dir():
            return []
        etats = [self.etat(f.stem) for f in sorted(self.dossier.glob("*.jsonl"))]
        return sorted(etats, key=lambda e: e["demarre_le"], reverse=True)
