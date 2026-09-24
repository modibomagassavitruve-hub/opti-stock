"""Journal des prédictions : chaque usage réel devient une donnée d'évaluation.

Répond au point aveugle de la mesure. Toutes les photos du catalogue ont été prises en rafale,
quelques secondes d'écart, même décor — impossible d'en tirer un jeu de test honnête pour la
question qui compte : « une photo prise un autre jour, sous un autre éclairage, retrouve-t-elle
la bonne monture ? ». Le seul moyen d'y répondre est de collecter ces photos à l'usage.

Chaque appel à /identifier écrit une ligne ici. Quand l'opticien indique ensuite la bonne
monture (POST /journal/<id>/choix), la ligne devient un cas de test étiqueté : photo réelle,
conditions réelles, vérité terrain donnée par un humain.

Format : JSONL en ajout seul, une ligne par prédiction, photo à côté. Pas de base de données --
le volume attendu est de quelques dizaines de lignes par jour, et un fichier se lit avec
n'importe quel outil. La table prediction_log du schéma prendra le relais le jour où le reste
du projet passera sur PostgreSQL.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, dossier: Path):
        self.dossier = Path(dossier)
        self.photos = self.dossier / "photos"
        self.lignes = self.dossier / "predictions.jsonl"

    def enregistrer(self, *, type_prediction: str, resultat: dict, version: str,
                    photo: bytes | None = None, latence_ms: int | None = None) -> str:
        """Écrit une prédiction et renvoie son identifiant, à retourner au client pour qu'il
        puisse ensuite déclarer la bonne réponse."""
        identifiant = uuid.uuid4().hex[:12]
        self.photos.mkdir(parents=True, exist_ok=True)
        if photo:
            (self.photos / f"{identifiant}.jpg").write_bytes(photo)

        ligne = {
            "id": identifiant,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": type_prediction,
            "version": version,
            "resultat": resultat,
            "latence_ms": latence_ms,
            "monture_choisie": None,   # rempli plus tard par l'opticien
        }
        with open(self.lignes, "a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        return identifiant

    def noter_choix(self, identifiant: str, monture: str) -> bool:
        """Déclare la bonne monture pour une prédiction. Écrit une ligne de correction plutôt
        que de réécrire le fichier : un journal en ajout seul ne peut pas être corrompu par
        une écriture concurrente, et l'historique des corrections reste lisible."""
        if not self.lignes.exists():
            return False
        if not any(json.loads(l).get("id") == identifiant for l in self._lire_brut()):
            return False
        with open(self.lignes, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": identifiant, "monture_choisie": monture,
                                "date": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
        return True

    def _lire_brut(self):
        if not self.lignes.exists():
            return []
        with open(self.lignes, encoding="utf-8") as f:
            return [l for l in f if l.strip()]

    def predictions(self) -> list[dict]:
        """Prédictions consolidées : les lignes de correction sont appliquées sur l'originale."""
        par_id: dict[str, dict] = {}
        for brut in self._lire_brut():
            ligne = json.loads(brut)
            ident = ligne.get("id")
            if ident in par_id:
                par_id[ident].update({k: v for k, v in ligne.items() if k != "date"})
            else:
                par_id[ident] = ligne
        return list(par_id.values())

    def bilan(self) -> dict:
        """recall@k en conditions réelles, sur les seules prédictions validées par un humain."""
        validees = [p for p in self.predictions()
                    if p.get("monture_choisie") and p.get("resultat", {}).get("resultats")]
        if not validees:
            return {"predictions": len(self.predictions()), "validees": 0}

        def dans_top(p, k):
            return p["monture_choisie"] in [r["monture"] for r in p["resultat"]["resultats"][:k]]

        fiables = [p for p in validees if p["resultat"].get("fiable")]
        return {
            "predictions": len(self.predictions()),
            "validees": len(validees),
            "recall@1": sum(dans_top(p, 1) for p in validees) / len(validees),
            "recall@5": sum(dans_top(p, 5) for p in validees) / len(validees),
            "marquees_fiables": len(fiables),
            "precision_si_fiable": (sum(dans_top(p, 1) for p in fiables) / len(fiables)
                                    if fiables else None),
        }


def chronometre():
    """Renvoie une fonction qui donne les millisecondes écoulées depuis son appel."""
    debut = time.perf_counter()
    return lambda: int((time.perf_counter() - debut) * 1000)
