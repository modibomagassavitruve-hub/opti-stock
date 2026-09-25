"""Réseau entre opticiens : trouver une monture chez un confrère et le contacter (module 3).

ATTENTION -- IDENTIFICATION DE DÉMONSTRATION
Le jeton remis à l'inscription suffit à agir au nom d'une boutique : il n'y a ni mot de passe,
ni vérification d'e-mail, ni contrôle du SIRET. C'est assumé pour montrer le produit, et
INSUFFISANT pour un service ouvert -- quiconque devine ou intercepte un jeton peut lire les
messages d'une boutique et écrire en son nom. Avant toute mise en service réelle, brancher une
authentification externe (le schéma la délègue explicitement) et reprendre ce module.

Ce qui circule ici : marque + référence, jamais l'identifiant local de monture. Le « 50 » d'une
boutique n'est pas celui d'une autre ; seul le couple marque/référence a un sens partagé. C'est
aussi pourquoi le partage est explicite, monture par monture : une boutique décide ce qu'elle
expose, le reste de son stock reste privé.

Stockage : un JSONL en ajout seul, comme le journal et l'inventaire.
"""
from __future__ import annotations

import json
import re
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _maintenant() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cle_monture(marque: str, reference: str) -> str:
    """Clé partagée entre boutiques. Les références sont gravées puis relues par OCR, d'où les
    confusions O/0 et I/1 : on normalise pour que « OS866 » et « 0S866 » se retrouvent."""
    brut = f"{marque}|{reference}".upper()
    return re.sub(r"[^A-Z0-9|]", "", brut).replace("O", "0").replace("I", "1")


class Reseau:
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
        complet = {**ligne, "date": _maintenant()}
        with open(self.fichier, "a", encoding="utf-8") as f:
            f.write(json.dumps(complet, ensure_ascii=False) + "\n")
        return complet

    # ------------------------------------------------------------- boutiques
    def inscrire(self, nom: str, ville: str, email: str) -> dict:
        boutique = uuid.uuid4().hex[:8]
        jeton = secrets.token_urlsafe(24)
        self._ajouter({"type": "boutique", "boutique": boutique, "nom": nom,
                        "ville": ville, "email": email, "jeton": jeton})
        return {"boutique": boutique, "nom": nom, "ville": ville, "jeton": jeton}

    # Ce que la liste des boutiques expose. Le jeton vaut mot de passe ; l'e-mail est une donnée
    # personnelle qu'aucun confrère n'a besoin de lire -- la mise en relation passe par la
    # messagerie interne, pas par la diffusion des coordonnées. Les deux restent dans le JSONL.
    _PUBLIC = ("boutique", "nom", "ville", "date")

    def boutiques(self) -> dict[str, dict]:
        """Annuaire public : ni jeton, ni e-mail. Ce dictionnaire alimente des réponses d'API."""
        return {l["boutique"]: {k: v for k, v in l.items() if k in self._PUBLIC}
                for l in self._lignes() if l["type"] == "boutique"}

    def authentifier(self, jeton: str) -> str | None:
        for l in self._lignes():
            if l["type"] == "boutique" and secrets.compare_digest(l["jeton"], jeton):
                return l["boutique"]
        return None

    # ---------------------------------------------------------------- partage
    def partager(self, boutique: str, montures: list[dict]) -> int:
        """`montures` : [{marque, reference, libelle}]. Remplace ce que la boutique expose."""
        propres = [m for m in montures if (m.get("marque") or "").strip()]
        self._ajouter({"type": "partage", "boutique": boutique, "montures": propres})
        return len(propres)

    def partage(self, boutique: str) -> list[dict]:
        """Ce que cette boutique propose actuellement au réseau."""
        return self._partages().get(boutique, [])

    def _partages(self) -> dict[str, list[dict]]:
        """Dernier partage déclaré par chaque boutique."""
        par_boutique: dict[str, list[dict]] = {}
        for l in self._lignes():
            if l["type"] == "partage":
                par_boutique[l["boutique"]] = l["montures"]
        return par_boutique

    def chercher(self, marque: str = "", reference: str = "",
                  sauf_boutique: str = "") -> list[dict]:
        """Qui expose une monture correspondante. La référence prime ; à défaut, la marque."""
        boutiques = self.boutiques()
        bloques = self._blocages()
        cible = cle_monture(marque, reference) if reference else ""
        resultats = []
        for boutique, montures in self._partages().items():
            if boutique == sauf_boutique or boutique not in boutiques:
                continue
            if sauf_boutique and (boutique, sauf_boutique) in bloques:
                continue   # elle a bloqué le demandeur : elle n'apparaît pas
            for m in montures:
                if cible:
                    if cle_monture(m.get("marque", ""), m.get("reference", "")) != cible:
                        continue
                elif marque and m.get("marque", "").strip().casefold() != marque.strip().casefold():
                    continue
                resultats.append({**m, "boutique": boutique,
                                   "nom": boutiques[boutique]["nom"],
                                   "ville": boutiques[boutique].get("ville", "")})
        return resultats

    # ---------------------------------------------------------- conversations
    def ouvrir_conversation(self, demandeuse: str, sollicitee: str, monture: str = "") -> str:
        if demandeuse == sollicitee:
            raise ValueError("Une boutique ne se contacte pas elle-même")
        if (sollicitee, demandeuse) in self._blocages():
            raise PermissionError("Cette boutique ne souhaite pas être contactée")

        for l in self._lignes():
            if (l["type"] == "conversation" and l["demandeuse"] == demandeuse
                    and l["sollicitee"] == sollicitee and l.get("monture", "") == monture):
                return l["conversation"]   # une seule conversation par (paire, monture)

        conversation = uuid.uuid4().hex[:8]
        self._ajouter({"type": "conversation", "conversation": conversation,
                        "demandeuse": demandeuse, "sollicitee": sollicitee, "monture": monture})
        return conversation

    def _conversation(self, conversation: str) -> dict | None:
        for l in self._lignes():
            if l["type"] == "conversation" and l["conversation"] == conversation:
                return l
        return None

    def ecrire(self, conversation: str, auteur: str, texte: str) -> dict:
        conv = self._conversation(conversation)
        if conv is None:
            raise KeyError(f"Conversation {conversation} inconnue")
        if auteur not in (conv["demandeuse"], conv["sollicitee"]):
            raise PermissionError("Cette boutique ne participe pas à cette conversation")
        autre = conv["sollicitee"] if auteur == conv["demandeuse"] else conv["demandeuse"]
        if (autre, auteur) in self._blocages():
            raise PermissionError("Cette boutique ne souhaite plus être contactée")
        if not texte.strip():
            raise ValueError("Message vide")
        return self._ajouter({"type": "message", "conversation": conversation,
                               "auteur": auteur, "texte": texte.strip()[:2000]})

    def messages(self, conversation: str, lecteur: str) -> list[dict]:
        conv = self._conversation(conversation)
        if conv is None:
            raise KeyError(f"Conversation {conversation} inconnue")
        if lecteur not in (conv["demandeuse"], conv["sollicitee"]):
            raise PermissionError("Cette boutique ne participe pas à cette conversation")
        return [{"auteur": l["auteur"], "texte": l["texte"], "date": l["date"]}
                for l in self._lignes()
                if l["type"] == "message" and l["conversation"] == conversation]

    def conversations(self, boutique: str) -> list[dict]:
        boutiques = self.boutiques()
        messages = [l for l in self._lignes() if l["type"] == "message"]
        sorties = []
        for l in self._lignes():
            if l["type"] != "conversation":
                continue
            if boutique not in (l["demandeuse"], l["sollicitee"]):
                continue
            interlocuteur = l["sollicitee"] if boutique == l["demandeuse"] else l["demandeuse"]
            siens = [m for m in messages if m["conversation"] == l["conversation"]]
            sorties.append({
                "conversation": l["conversation"],
                "interlocuteur": boutiques.get(interlocuteur, {}).get("nom", interlocuteur),
                "monture": l.get("monture", ""),
                "messages": len(siens),
                "dernier": siens[-1]["texte"][:80] if siens else "",
                "date": siens[-1]["date"] if siens else l["date"],
            })
        return sorted(sorties, key=lambda c: c["date"], reverse=True)

    # ------------------------------------------------------- modération
    def bloquer(self, boutique: str, bloquee: str) -> None:
        """Une boutique bloquée ne peut plus ouvrir de conversation ni écrire, et n'apparaît
        plus dans les recherches du bloqué."""
        if boutique == bloquee:
            raise ValueError("Une boutique ne se bloque pas elle-même")
        self._ajouter({"type": "blocage", "boutique": boutique, "bloquee": bloquee})

    def debloquer(self, boutique: str, bloquee: str) -> None:
        self._ajouter({"type": "deblocage", "boutique": boutique, "bloquee": bloquee})

    def _blocages(self) -> set[tuple[str, str]]:
        """Paires (bloqueur, bloquée) actives, les déblocages annulant les blocages."""
        actifs: set[tuple[str, str]] = set()
        for l in self._lignes():
            paire = (l.get("boutique"), l.get("bloquee"))
            if l["type"] == "blocage":
                actifs.add(paire)
            elif l["type"] == "deblocage":
                actifs.discard(paire)
        return actifs

    def signaler(self, conversation: str, auteur: str, motif: str) -> dict:
        return self._ajouter({"type": "signalement", "conversation": conversation,
                               "auteur": auteur, "motif": motif.strip()[:500]})

    def signalements(self) -> list[dict]:
        return [l for l in self._lignes() if l["type"] == "signalement"]
