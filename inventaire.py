"""Inventaire annuel : compter le rayon et le confronter au stock théorique (module 2).

Différence de nature avec /identifier, qui commande la conception : ici la monture est
forcément au catalogue, et l'opticien la tient en main. Ce n'est pas une identification mais
une CONFIRMATION -- la reconnaissance visuelle propose, l'humain valide d'un geste. Une
reconnaissance imparfaite reste donc utilisable, là où elle serait gênante pour une commande
fournisseur.

Ce que l'inventaire produit de précieux n'est pas la liste de ce qu'on a trouvé, mais l'ÉCART :
combien il devrait y en avoir, combien il y en a, et de combien on s'est trompé. Une version
antérieure ne savait dire que « vue / pas vue », faute de quantité théorique à comparer --
c'est le stock qui la fournit désormais.

Clôturer applique les comptages au stock : c'est le sens d'un récolement, le rayon fait foi.
Chaque correction passe par Stock.ajuster, qui en garde la trace et l'écart.

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

    def demarrer(self, theorique: dict[str, int], libelle: str = "") -> str:
        """Ouvre une session en figeant le stock théorique : il peut bouger pendant
        l'inventaire (une vente, une réception) sans fausser la comparaison finale."""
        session = uuid.uuid4().hex[:8]
        self._ajouter(session, {"type": "ouverture", "libelle": libelle,
                                 "theorique": {str(m): int(q) for m, q in theorique.items()}})
        return session

    def compter(self, session: str, monture: str, quantite: int = 1) -> dict:
        lignes = self._lignes(session)
        if not lignes:
            raise KeyError(f"Inventaire {session} inconnu")
        if self._statut(lignes) == "clos":
            raise ValueError(f"Inventaire {session} déjà clos")
        if quantite < 1:
            raise ValueError("Un comptage porte sur au moins une monture")
        self._ajouter(session, {"type": "comptage", "monture": monture, "quantite": quantite})
        return self.etat(session)

    def annuler_comptage(self, session: str, monture: str, quantite: int = 1) -> dict:
        """Retire un comptage. On ajoute une ligne d'annulation plutôt que d'en effacer une :
        le fichier reste en ajout seul, et l'historique des corrections reste lisible."""
        if not self._lignes(session):
            raise KeyError(f"Inventaire {session} inconnu")
        self._ajouter(session, {"type": "annulation", "monture": monture,
                                 "quantite": max(1, quantite)})
        return self.etat(session)

    def cloturer(self, session: str, stock=None) -> dict:
        """Clôt la session et, si un stock est fourni, y applique les quantités comptées.

        C'est le sens du récolement : le rayon fait foi. Les montures non comptées passent à
        zéro -- ne pas les toucher laisserait le stock affirmer qu'elles sont là alors qu'on
        vient de constater le contraire.
        """
        lignes = self._lignes(session)
        if not lignes:
            raise KeyError(f"Inventaire {session} inconnu")
        if self._statut(lignes) == "clos":
            raise ValueError(f"Inventaire {session} déjà clos")

        etat = self.etat(session)
        if stock is not None:
            for ligne in etat["lignes"]:
                if ligne["ecart"]:
                    stock.ajuster(ligne["monture"], ligne["comptee"],
                                   motif=f"inventaire {etat['libelle'] or session}")

        self._ajouter(session, {"type": "cloture", "corrigees": sum(
            1 for l in etat["lignes"] if l["ecart"]) if stock is not None else 0})
        return self.etat(session)

    @staticmethod
    def _statut(lignes: list[dict]) -> str:
        return "clos" if any(l["type"] == "cloture" for l in lignes) else "en_cours"

    def etat(self, session: str) -> dict:
        lignes = self._lignes(session)
        if not lignes:
            raise KeyError(f"Inventaire {session} inconnu")

        theorique: dict[str, int] = dict(lignes[0].get("theorique", {}))
        comptees: dict[str, int] = {}
        for l in lignes:
            if l["type"] == "comptage":
                comptees[l["monture"]] = comptees.get(l["monture"], 0) + l.get("quantite", 1)
            elif l["type"] == "annulation":
                comptees[l["monture"]] = max(0, comptees.get(l["monture"], 0)
                                              - l.get("quantite", 1))

        detail = []
        for monture in sorted(set(theorique) | set(comptees)):
            attendue, comptee = theorique.get(monture, 0), comptees.get(monture, 0)
            detail.append({"monture": monture, "theorique": attendue, "comptee": comptee,
                            "ecart": comptee - attendue})

        manquantes = [l for l in detail if l["ecart"] < 0]
        en_trop = [l for l in detail if l["ecart"] > 0]
        return {
            "session": session,
            "libelle": lignes[0].get("libelle", ""),
            "statut": self._statut(lignes),
            "demarre_le": lignes[0]["date"],
            "references_attendues": sum(1 for q in theorique.values() if q > 0),
            "pieces_attendues": sum(theorique.values()),
            "pieces_comptees": sum(comptees.values()),
            "references_comptees": sum(1 for q in comptees.values() if q > 0),
            # Les deux listes qui font l'intérêt de l'inventaire : ce qui manque, et ce qui est
            # là sans être au stock (retours non saisis, erreurs de rangement, vols rendus).
            "manquantes": manquantes,
            "en_trop": en_trop,
            "hors_stock": sorted(l["monture"] for l in detail if l["theorique"] == 0
                                  and l["comptee"] > 0),
            "lignes": detail,
        }

    def sessions(self) -> list[dict]:
        if not self.dossier.is_dir():
            return []
        etats = [self.etat(f.stem) for f in sorted(self.dossier.glob("*.jsonl"))]
        return sorted(etats, key=lambda e: e["demarre_le"], reverse=True)
