"""Le stock réel de la boutique : combien de chaque monture, et où (module 1 du schéma).

C'est le socle qui manquait. L'identification par photo dit *quelle* monture ; sans stock elle
ne débouche sur rien. L'inventaire comptait « vue / pas vue » faute de quantité théorique à
comparer. Et le réseau proposait « j'ai ce modèle » au lieu de « il m'en reste », ce qui fait
déplacer un confrère pour rien.

Journal de mouvements, pas compteur. Un compteur qu'on incrémente répond « il en reste 2 » ;
un journal répond « il en reste 2 parce que j'en ai reçu 5, vendu 2, et que l'inventaire en a
retrouvé 1 de moins ». En stock c'est la seule forme utile : les écarts sont la norme, et sans
historique on ne peut ni les expliquer ni les corriger de façon sûre.

Conséquence assumée : `sortir` refuse de descendre sous zéro. Un stock négatif n'existe pas en
rayon, et l'accepter rendrait tous les chiffres douteux. Quand le stock théorique est faux, on
le corrige par `ajuster`, qui pose une quantité constatée et laisse une trace de la correction.

Stockage : un JSONL en ajout seul, comme le journal, l'inventaire et le réseau.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MOTIFS_ENTREE = ("reception", "retour_client", "correction")
MOTIFS_SORTIE = ("vente", "casse", "perte", "retour_fournisseur", "correction")


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stock:
    def __init__(self, fichier: Path):
        self.fichier = Path(fichier)

    # ------------------------------------------------------------------ socle
    def _lignes(self) -> list[dict]:
        if not self.fichier.exists():
            return []
        with open(self.fichier, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _ajouter(self, ligne: dict) -> dict:
        self.fichier.parent.mkdir(parents=True, exist_ok=True)
        complet = {**ligne, "date": _horodatage()}
        with open(self.fichier, "a", encoding="utf-8") as f:
            f.write(json.dumps(complet, ensure_ascii=False) + "\n")
        return complet

    # -------------------------------------------------------------- mouvements
    def entrer(self, monture: str, quantite: int = 1, emplacement: str = "",
                motif: str = "reception") -> dict:
        if quantite < 1:
            raise ValueError("Une entrée porte sur au moins une monture")
        if motif not in MOTIFS_ENTREE:
            raise ValueError(f"Motif d'entrée inconnu : {motif}")
        self._ajouter({"type": "entree", "monture": monture, "quantite": quantite,
                        "emplacement": emplacement.strip(), "motif": motif})
        return self.etat_monture(monture)

    def sortir(self, monture: str, quantite: int = 1, motif: str = "vente") -> dict:
        """Refuse de descendre sous zéro : voir l'en-tête du module."""
        if quantite < 1:
            raise ValueError("Une sortie porte sur au moins une monture")
        if motif not in MOTIFS_SORTIE:
            raise ValueError(f"Motif de sortie inconnu : {motif}")
        disponible = self.quantite(monture)
        if quantite > disponible:
            raise ValueError(f"{monture} : {disponible} en stock, sortie de {quantite} refusée")
        self._ajouter({"type": "sortie", "monture": monture, "quantite": quantite,
                        "motif": motif})
        return self.etat_monture(monture)

    def ajuster(self, monture: str, quantite: int, motif: str = "inventaire") -> dict:
        """Pose une quantité constatée, quel que soit le théorique. C'est ce que produit un
        inventaire : le rayon fait foi, l'écart est enregistré tel quel."""
        if quantite < 0:
            raise ValueError("Une quantité constatée ne peut pas être négative")
        avant = self.quantite(monture)
        self._ajouter({"type": "ajustement", "monture": monture, "quantite": quantite,
                        "motif": motif, "ecart": quantite - avant})
        return self.etat_monture(monture)

    def deplacer(self, monture: str, emplacement: str) -> dict:
        self._ajouter({"type": "emplacement", "monture": monture,
                        "emplacement": emplacement.strip()})
        return self.etat_monture(monture)

    # ------------------------------------------------------------------- états
    def etat(self) -> dict[str, dict]:
        """Quantité et emplacement de chaque monture ayant connu un mouvement.

        Les montures à zéro sont conservées : « j'en avais, je n'en ai plus » n'est pas la même
        information que « je n'en ai jamais eu », notamment pour recommander chez un confrère.
        """
        etats: dict[str, dict] = {}
        for l in self._lignes():
            e = etats.setdefault(l["monture"], {"monture": l["monture"], "quantite": 0,
                                                 "emplacement": "", "maj_le": l["date"]})
            if l["type"] == "entree":
                e["quantite"] += l["quantite"]
            elif l["type"] == "sortie":
                e["quantite"] -= l["quantite"]
            elif l["type"] == "ajustement":
                e["quantite"] = l["quantite"]
            if l.get("emplacement"):
                e["emplacement"] = l["emplacement"]
            e["maj_le"] = l["date"]
        return etats

    def etat_monture(self, monture: str) -> dict:
        return self.etat().get(monture, {"monture": monture, "quantite": 0,
                                          "emplacement": "", "maj_le": ""})

    def quantite(self, monture: str) -> int:
        return self.etat_monture(monture)["quantite"]

    def en_stock(self) -> dict[str, dict]:
        """Uniquement ce qui est réellement en rayon -- ce que le réseau a le droit d'exposer."""
        return {m: e for m, e in self.etat().items() if e["quantite"] > 0}

    def mouvements(self, monture: str = "") -> list[dict]:
        lignes = self._lignes()
        return [l for l in lignes if not monture or l["monture"] == monture]

    def bilan(self) -> dict:
        etats = self.etat()
        return {
            "references": len(etats),
            "references_en_stock": sum(1 for e in etats.values() if e["quantite"] > 0),
            "pieces": sum(e["quantite"] for e in etats.values()),
            "epuisees": sorted(m for m, e in etats.items() if e["quantite"] == 0),
            "mouvements": len(self._lignes()),
        }
